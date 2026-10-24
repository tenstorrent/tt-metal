# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The example inputs a model publishes about ITSELF, so a batch need not be invented.

WHY THIS EXISTS. `_batch_prompt_block` told the builder to "feed {batch} DISTINCT reference inputs".
DISTINCT was the only constraint, so the builder authored them. A Qwen-Image-Edit bring-up ran 32
prompts the agent wrote; every sample that later missed the PCC bar was one of the written ones, and
because each sample varied its content AND its seed at once, a miss could not be attributed to
either. An input nobody can source makes a PCC number unciteable: it reads the same as a hardware
fault, and re-running never settles it.

WHAT IT READS -- only what the model ships or declares about itself:
  * the model card's ```python fences;
  * the docstrings beside the class the model NAMES as its entry point (the pipeline index's
    `_class_name`, else the config's `architectures`), including module-level example strings.
Everything is parsed with `ast`. Nothing is executed -- these documents come from the hub, and the
point is to read an example, not to run one.

WHAT IT DOES NOT DO. No kwarg name is typed in this module: names come from the example itself and,
for positional arguments, from the entry point's own signature. The only identifiers here are the
ecosystem's own loading/seeding APIs (`from_pretrained`, `manual_seed`), which are library calls, not
model or stage names -- a model that renames its stages tomorrow still parses.

`discover(model_id)` returns `ExampleInputs` or None. None means the model publishes no parseable
example, and the caller must SAY so rather than let an invented input pass as a sourced one.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .probe import COMPOSITE_INDEX_FILE, ROOT_CONFIG_FILE, fetch_repo_json, fetch_repo_text

# The hub's model-card document. Named here for the same reason probe.py names its config files:
# a protocol filename belongs in a constant, not inline at the call site.
CARD_FILE = "README.md"

# Keys a checkpoint uses to name its own entry-point class.
_ARCHITECTURES_KEY = "architectures"
_CLASS_NAME_KEY = "_class_name"

# A document that records which library wrote it does so in a "<library>_version" key
# (`_diffusers_version`, `transformers_version`, ...). The library name is read OFF that key, so no
# library name is hardcoded here.
_VERSION_KEY_RE = re.compile(r"^_?(?P<lib>[a-z][a-z0-9_]*)_version$")

_FENCE_RE = re.compile(r"```(?:py|python)[^\n]*\n(.*?)```", re.S)
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_DOCTEST_RE = re.compile(r"^\s*(?:>>>|\.\.\.) ?", re.M)

# Ecosystem APIs, not model names: the call that builds the entry point, and the call that seeds it.
_LOAD_SUFFIX = "from_pretrained"
_SEED_SUFFIX = "manual_seed"

_URL_PREFIXES = ("http://", "https://")


@dataclass(frozen=True)
class Resource:
    """A value the example LOADS rather than spells out: `load_image(url)`, `Image.open(path)`."""

    call: str
    args: Tuple[Any, ...]

    @property
    def target(self) -> Optional[str]:
        for a in self.args:
            if isinstance(a, str):
                return a
        return None

    def resolves(self) -> bool:
        """True when the thing it names can actually be fetched. A card that opens a relative file
        it never ships (`Image.open("./input.png")`) publishes an example nobody else can run."""
        t = self.target
        if not t:
            return False
        return t.startswith(_URL_PREFIXES) or os.path.exists(t)

    def __str__(self) -> str:
        return f"{self.call}({', '.join(repr(a) for a in self.args)})"


@dataclass(frozen=True)
class Seed:
    """A seeded generator the example declares."""

    value: int

    def __str__(self) -> str:
        return f"seed {self.value}"


@dataclass
class ExampleInputs:
    """One published example call, as literals."""

    kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    sources: Tuple[str, ...] = ()

    @property
    def resources(self) -> List[Resource]:
        return [v for v in self.kwargs.values() if isinstance(v, Resource)]

    @property
    def resolvable(self) -> bool:
        """Every resource it names can be fetched. An example with no resources is trivially so."""
        return all(r.resolves() for r in self.resources)

    def describe(self) -> str:
        lines = [f"  {k} = {_fmt(v)}" for k, v in self.kwargs.items()]
        if self.seed is not None:
            lines.append(f"  (seed the example declares: {self.seed})")
        lines.append("  source: " + "; ".join(self.sources or ("unrecorded",)))
        if not self.resolvable:
            unresolved = [str(r) for r in self.resources if not r.resolves()]
            lines.append(
                "  NOT SELF-CONTAINED -- these name something the repo does not ship: " + ", ".join(unresolved)
            )
        return "\n".join(lines)


def _fmt(val: Any) -> str:
    """Render a value the way the example spells it, resources included at any nesting depth."""
    if isinstance(val, Resource):
        return str(val)
    if isinstance(val, list):
        return "[" + ", ".join(_fmt(v) for v in val) + "]"
    if isinstance(val, dict):
        return "{" + ", ".join(f"{k!r}: {_fmt(v)}" for k, v in val.items()) + "}"
    return repr(val)


def _card(model_id: str) -> Optional[str]:
    return fetch_repo_text(model_id, CARD_FILE)


def _declared_libraries(model_id: str, card: Optional[str]) -> List[str]:
    """Modules the model itself says it was written by, most specific first."""
    libs: List[str] = []
    if card:
        fm = _FRONTMATTER_RE.match(card)
        if fm:
            for line in fm.group(1).splitlines():
                key, _, val = line.partition(":")
                if key.strip() == "library_name" and val.strip():
                    libs.append(val.strip())
    for doc_name in (COMPOSITE_INDEX_FILE, ROOT_CONFIG_FILE):
        doc = fetch_repo_json(model_id, doc_name)
        for key in doc or {}:
            m = _VERSION_KEY_RE.match(str(key))
            if m:
                libs.append(m.group("lib"))
    seen, out = set(), []
    for lib in libs:
        if lib not in seen:
            seen.add(lib)
            out.append(lib)
    return out


def _declared_class(model_id: str, card: Optional[str]):
    """The entry-point class the model NAMES for itself, imported from the library it names.

    Read from the pipeline index's `_class_name`, else the config's `architectures` -- never guessed
    from the model id, the folder name or a tag."""
    names: List[str] = []
    index = fetch_repo_json(model_id, COMPOSITE_INDEX_FILE) or {}
    if isinstance(index.get(_CLASS_NAME_KEY), str):
        names.append(index[_CLASS_NAME_KEY])
    cfg = fetch_repo_json(model_id, ROOT_CONFIG_FILE) or {}
    arch = cfg.get(_ARCHITECTURES_KEY)
    if isinstance(arch, list):
        names += [a for a in arch if isinstance(a, str)]
    for lib in _declared_libraries(model_id, card):
        try:
            mod = importlib.import_module(lib)
        except Exception:
            continue
        for name in names:
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls
    return None


def _fences(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [_DOCTEST_RE.sub("", f) for f in _FENCE_RE.findall(text)]


def _class_fences(cls) -> List[str]:
    """Examples published beside the class: its own docstring, its module's, and any module-level
    string constant holding a fence (the docstring-constant convention several libraries use).

    The constant is found by looking for a fence in it, never by its NAME, so a library that calls
    it something else is read just the same."""
    if cls is None:
        return []
    texts: List[str] = [inspect.getdoc(cls) or ""]
    mod = inspect.getmodule(cls)
    if mod is not None:
        texts.append(getattr(mod, "__doc__", "") or "")
        for name in dir(mod):
            if name.startswith("__"):
                continue
            try:
                val = getattr(mod, name)
            except Exception:
                continue
            if isinstance(val, str) and "```" in val:
                texts.append(val)
    out: List[str] = []
    for t in texts:
        out += _fences(t)
    return out


def _dotted(node) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _value(node, env: Dict[str, Any]) -> Any:
    """Resolve one AST node to a literal, a Resource, a Seed, or `_UNKNOWN`. Never executes."""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return env.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            key = _value(k, env) if k is not None else None
            val = _value(v, env)
            if isinstance(key, str) and val is not _UNKNOWN:
                out[key] = val
        return out
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [_value(e, env) for e in node.elts]
        return [v for v in vals if v is not _UNKNOWN]
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name.split(".")[-1] == _SEED_SUFFIX:
            args = [_value(a, env) for a in node.args]
            ints = [a for a in args if isinstance(a, int) and not isinstance(a, bool)]
            return Seed(ints[0]) if ints else _UNKNOWN
        # A chained call (`Image.open(p).convert("RGB")`) carries the thing being loaded in the
        # INNER call; the outer link only transforms it. Report the inner one, or the resource would
        # be recorded as its own colour-space argument.
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            inner = _value(node.func.value, env)
            if isinstance(inner, Resource):
                return inner
        args = tuple(a for a in (_value(a, env) for a in node.args) if a is not _UNKNOWN)
        return Resource(name, args) if args else _UNKNOWN
    return _UNKNOWN


class _Unknown:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unresolved>"


_UNKNOWN = _Unknown()


def _positional_names(cls, entry_attr: Optional[str], count: int) -> List[Optional[str]]:
    """Names for positional arguments, taken from the entry point's OWN signature."""
    if cls is None or count <= 0:
        return [None] * count
    target = getattr(cls, entry_attr, None) if entry_attr else getattr(cls, "__call__", None)
    try:
        params = [
            p.name
            for p in inspect.signature(target).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"
        ]
    except Exception:
        return [None] * count
    return [params[i] if i < len(params) else None for i in range(count)]


def _parse_fence(src: str, cls, origin: str) -> Optional[ExampleInputs]:
    """Extract the literal inputs of the example's entry call(s). None if it invokes nothing loaded.

    An example loads more than one object (a model AND its tokenizer/processor), and the inputs are
    spread across the calls it makes on them. Where the loaded object can be tied to the class the
    model declares -- its loader is spelled with that class's own name -- that object's richest call
    is THE entry call and its positional arguments are named from that class's signature. Where it
    cannot (the example loads via an `Auto*` factory, so the spelled name is not the model's class),
    naming positionals from the declared class would attach one object's parameter names to another
    object's call, so they are labelled by position instead and every loaded object's literal
    keywords are merged. Guessing is the one thing that is never done."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    env: Dict[str, Any] = {}
    loaders: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if isinstance(node.value, ast.Call):
            dotted = _dotted(node.value.func)
            if dotted.split(".")[-1] == _LOAD_SUFFIX:
                loaders[name] = dotted.rsplit(".", 1)[0]
                continue
        val = _value(node.value, env)
        if val is not _UNKNOWN:
            env[name] = val
    if not loaders:
        return None

    # Every seed the fence declares, wherever it sits: a bare statement, an assignment, or a value
    # inside the mapping the call is given.
    seeds: List[int] = [
        v.value for n in ast.walk(tree) if isinstance(n, ast.Call) for v in (_value(n, env),) if isinstance(v, Seed)
    ]

    cls_name = getattr(cls, "__name__", None)
    cls_var = next((var for var, base in loaders.items() if cls_name and base == cls_name), None)

    calls: List[Tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in loaders:
            calls.append((node.func.id, node))
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in loaders:
                calls.append((node.func.value.id, node))
    if not calls:
        return None

    def _weight(item) -> int:
        return len(item[1].args) + len(item[1].keywords)

    # The declared class's richest call leads (its keywords and its signature's names win ties), but
    # the other loaded objects still contribute: for a model whose example tokenises or pre-processes
    # first, the CONTENT being fed lives in that call and the model's own call sees only generation
    # settings. Dropping it would report an example with no inputs in it.
    if cls_var:
        primary = max((c for c in calls if c[0] == cls_var), key=_weight)
        chosen = [primary] + [c for c in calls if c is not primary]
    else:
        chosen = calls

    kwargs: Dict[str, Any] = {}
    for var, call in chosen:
        for kw in call.keywords:
            val = _value(kw.value, env)
            if val is _UNKNOWN:
                continue
            if kw.arg is None:  # **mapping
                if isinstance(val, dict):
                    kwargs.update(val)
                continue
            kwargs.setdefault(kw.arg, val)
        pos = [_value(a, env) for a in call.args]
        if var == cls_var:
            attr = call.func.attr if isinstance(call.func, ast.Attribute) else None
            names = _positional_names(cls, attr, len(pos))
        else:
            label = _dotted(call.func)
            names = [f"{label}[{i}]" for i in range(len(pos))]
        for name, val in zip(names, pos):
            if name and val is not _UNKNOWN:
                kwargs.setdefault(name, val)

    for key, val in list(kwargs.items()):
        if isinstance(val, Seed):
            kwargs.pop(key)
    if not kwargs and not seeds:
        return None
    return ExampleInputs(kwargs=kwargs, seed=seeds[0] if seeds else None, sources=(origin,))


def discover(model_id: str) -> Optional[ExampleInputs]:
    """The model's own published example inputs, or None if it publishes none that parse.

    Where several examples exist they are ranked self-contained first (an example naming a file the
    repo does not ship cannot be reproduced), then by how much of the call they pin down. A seed
    declared by any example is carried onto the winner even when the winner declares none -- the two
    are independent facts, and the merge is recorded in `sources` rather than implied to be one
    document."""
    card = _card(model_id)
    cls = _declared_class(model_id, card)
    candidates: List[ExampleInputs] = []
    for i, src in enumerate(_fences(card)):
        got = _parse_fence(src, cls, f"{CARD_FILE} example {i + 1}")
        if got:
            candidates.append(got)
    if cls is not None:
        where = f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__name__', '?')} docstring"
        for i, src in enumerate(_class_fences(cls)):
            got = _parse_fence(src, cls, f"{where} example {i + 1}")
            if got:
                candidates.append(got)
    if not candidates:
        return None
    best = max(candidates, key=lambda e: (e.resolvable, len(e.kwargs)))
    if best.seed is None:
        for other in candidates:
            if other.seed is not None:
                best = ExampleInputs(
                    kwargs=best.kwargs, seed=other.seed, sources=best.sources + (f"seed from {other.sources[0]}",)
                )
                break
    return best
