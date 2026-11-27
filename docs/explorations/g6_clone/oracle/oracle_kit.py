"""Helpers for the oracle of an exploration dossier. Torch only, no ttnn, no device.

Copied into every dossier so that the oracle runs wherever the dossier goes:

    from oracle_kit import Checks, load_upstream, max_rel_diff, metrics, to_bfloat16, bfloat8_roundtrip, tf32

What is exact and what is an estimate:
- metrics() uses the definitions of eval/metrics.py (PCC and relative RMS, float64 accumulation).
- to_bfloat16() is the storage rounding of the host (nearest even), exact.
- bfloat8_roundtrip() re-implements the host packing rule of block floats (one shared exponent
  per 16 consecutive datums of a tile row). It models storage, not device arithmetic.
- tf32() is the cut a float32 operand takes on its way into the matrix engine (10 mantissa bits).
- Nothing here models the arithmetic of the matrix engine, the vector engine, accumulation in a
  16-bit DEST, the device packer, denormal flush or NaN handling (torch keeps denormals, NaN
  and -0). Quote every figure as "emulated (storage and operands only), not confirmed on a
  device". Only the bfloat8_b cell of Checks.cell() is a floor; see its docstring.

What a run leaves behind (Checks.finish() writes oracle_report.json):
- "checks": one row per check with its kind (equivalence, probe, reformulation, precision) and
  status. "pass" and "FAIL" belong to equivalence checks, probes and reformulations; "skipped"
  carries its reason; a precision row has status "info", its detail ends with the mark above,
  it never fails a run and it is not counted as a passed check. An equivalence check that ran
  also carries "evidence", the level of that one comparison.
- "evidence" (E1 to E4): the headline, the level reached for the implementation the request is
  about. "evidence_levels": the levels of the passed equivalence checks, sorted.
  "evidence_scope": one sentence, which implementation each level covers and what did not run.
- "max_rel_diff" (the largest difference of the equivalence checks, whatever their level; null
  when none ran) and "compared_in" (the float type both sides of them were computed in).
- "notes": remarks that never fail a run, a list of sentences, each also printed as a line
  "note: ..." after the summary. Empty, or the one remark there is: compared_in is float64 and
  the largest difference lies above 1e-12 and within the gate of 1e-6. That is the size of
  float32 rounding, so a float32 cast may be left on upstream's call path where no dtype shows
  it (.float() ... .type_as(x) hands back the dtype of the input).
- "upstream_edits": one entry per load_upstream() call: the pinned file and its sha256, the
  literal replacements with the number of places each one changed, the modules that were
  stand-ins and the names that were injected. Empty when upstream ran as it is pinned.
- "files": the sha256 of every .py file beside the report. Hashes that differ from the files
  mean that another version of the oracle wrote the report (check_dossier.py compares them):
  rerun check_oracle.py after every edit.
- "passed": no row has status FAIL. finish() adds such a row itself when the evidence levels
  and the comparisons that ran disagree (see Checks).
"""

import builtins
import datetime
import hashlib
import importlib
import io
import json
import math
import os
import re
import sys
import tokenize
import types

import torch

# (PCC >=, relative RMS <=) per (input dtype, fp32_dest_acc_en) at HiFi4, copied from
# eval/metrics.py DEFAULT_TOLERANCES at tt_ops_code_gen b93830b. The live file wins.
DEFAULT_TOLERANCES = {
    ("float32", True): (0.999, 0.01),
    ("bfloat16", True): (0.995, 0.04),
    ("bfloat16", False): (0.99, 0.13),
    ("bfloat8_b", True): (0.99, 0.10),
    ("bfloat8_b", False): (0.99, 0.13),
}
EQUIVALENCE_GATE = 1e-6  # float64 math-identity gate used by the golden-tests skill
# Two float64 computations of one formula stay below FLOAT64_SIZED as a rule. A largest difference above it that
# still passes the gate has the size of float32 rounding: Checks.finish() then says FLOAT32_SIZED_NOTE.
FLOAT64_SIZED = 1e-12
FLOAT32_SIZED_NOTE = ("the largest difference is float32-sized: a float32 cast may be left on upstream's call path "
                      "(.float() ... .type_as(x)); replace it through load_upstream(replace=...), or set "
                      'checks.compared_in = "float32" when upstream computes only there')
TILE = 32
BLOCK_FLOAT_GROUP = 16
MARK = "emulated (storage and operands only), not confirmed on a device"  # ends every precision row
EVIDENCE_LEVELS = ("E1", "E2", "E3", "E4")
COMPARISON_LEVELS = EVIDENCE_LEVELS[:3]  # what a comparison that ran can be; E4 says that none did
# Float types an equivalence check may be computed in, each with the types that are narrower.
NARROWER = {"float64": ("float32", "bfloat16", "float16"), "float32": ("bfloat16", "float16")}
BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the bundle directory: this file lies in its oracle/
_UPSTREAM_LOADS = []  # one record per load_upstream() call; Checks.finish() writes them as "upstream_edits"


def seeded(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def max_rel_diff(actual, expected):
    """Largest absolute difference relative to the largest reference magnitude (float64)."""
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)
    if actual.shape != expected.shape:
        return math.inf
    if expected.numel() == 0:
        return 0.0
    both_nan = torch.isnan(actual) & torch.isnan(expected)
    same_inf = torch.isinf(actual) & torch.isinf(expected) & (torch.sign(actual) == torch.sign(expected))
    difference = torch.where(both_nan | same_inf, torch.zeros_like(actual), (actual - expected).abs())
    if not bool(torch.isfinite(difference).all()):
        return math.inf
    finite = expected[torch.isfinite(expected)].abs()
    scale = finite.max().item() if finite.numel() else 0.0
    return difference.max().item() / max(scale, 1e-300)


def metrics(actual, expected):
    """PCC and relative RMS as the golden suites compute them (eval/metrics.py compute_metrics_torch)."""
    actual = actual.to(torch.float32).flatten()
    expected = expected.to(torch.float32).flatten()
    count = actual.numel()
    actual_centered = actual - (actual.sum(dtype=torch.float64) / count).to(actual.dtype)
    expected_centered = expected - (expected.sum(dtype=torch.float64) / count).to(expected.dtype)
    covariance = (actual_centered * expected_centered).sum(dtype=torch.float64)
    denominator = torch.sqrt(
        actual_centered.pow(2).sum(dtype=torch.float64) * expected_centered.pow(2).sum(dtype=torch.float64)
    )
    pcc = (covariance / denominator).item()
    if math.isnan(pcc):
        pcc = 1.0
    difference = actual - expected
    abs_rms = math.sqrt((difference.pow(2).sum(dtype=torch.float64) / count).item())
    spread = expected.std().item() if count > 1 else 0.0
    return {
        "pcc": pcc,
        "rms": abs_rms / spread if spread > 1e-12 else abs_rms,
        "max_abs_diff": difference.abs().max().item(),
        "has_inf": bool(torch.isinf(actual).any()),
        "has_nan": bool(torch.isnan(actual).any()),
    }


def _limits(dtype, fp32_dest_acc_en, tolerance):
    """The (PCC, relative RMS) pair a figure is held against: `tolerance`, else the default row."""
    if tolerance is not None:
        return tuple(tolerance)
    key = (dtype, bool(fp32_dest_acc_en))
    if key in DEFAULT_TOLERANCES:
        return DEFAULT_TOLERANCES[key]
    if key == ("float32", False):
        raise ValueError("no default tolerance for float32 with fp32_dest_acc_en=False: float32 with a 16-bit DEST is the cell "
                         "every op must refuse; a figure for it needs tolerance=(pcc, rms)")
    raise ValueError(f"no default tolerance for dtype {dtype} (the table covers float32, bfloat16 and bfloat8_b): pass tolerance=(pcc, rms)")


def within(measured, dtype, fp32_dest_acc_en=True, tolerance=None):
    """Does a metrics() result meet the (PCC, relative RMS) row of the tolerance table?

    A cell without a default row (float32 with a 16-bit DEST, bfloat4_b, integers) needs
    tolerance=(pcc, rms): without it a ValueError says so.
    """
    pcc_limit, rms_limit = _limits(dtype, fp32_dest_acc_en, tolerance)
    return measured["pcc"] >= pcc_limit and measured["rms"] <= rms_limit and not measured["has_inf"] and not measured["has_nan"]


def to_bfloat16(tensor):
    """float32 -> bfloat16 -> float32 with the host's rounding (nearest, ties to even)."""
    return tensor.to(torch.float32).to(torch.bfloat16).to(torch.float32)


def bfloat8_roundtrip(tensor, magnitude_bits=7):
    """Store as bfloat8_b (magnitude_bits=7) or bfloat4_b (3) the way the host packs it, and read back.

    One 8-bit exponent is shared by 16 consecutive datums along the last dimension (a face row of
    a tile); each datum keeps a sign and `magnitude_bits` bits aligned to the largest exponent of
    its group, rounded to nearest even and clamped. A datum at or below 1/128 of the group's power
    of two (bfloat4_b: 1/8) becomes zero, so does every datum below 1/256 (1/16) of the group
    maximum. An inf sets the exponent of its group and erases its neighbours; a NaN comes back as
    inf here. The last dimension is padded with zeros to a multiple of 16.
    """
    values = tensor.to(torch.float32).contiguous()
    shape = values.shape
    width = shape[-1]
    padded_width = -(-width // BLOCK_FLOAT_GROUP) * BLOCK_FLOAT_GROUP
    if padded_width != width:
        values = torch.nn.functional.pad(values, (0, padded_width - width))
    bits = values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = ((bits >> 31) & 1).reshape(-1, BLOCK_FLOAT_GROUP)
    exponent = ((bits >> 23) & 0xFF).reshape(-1, BLOCK_FLOAT_GROUP)
    mantissa = ((bits & 0x7FFFFF) | 0x800000).reshape(-1, BLOCK_FLOAT_GROUP)
    shared = exponent.max(dim=1, keepdim=True).values
    mantissa = mantissa >> (shared - exponent).clamp(max=40)
    shift = 24 - magnitude_bits
    remainder = mantissa & ((1 << shift) - 1)
    magnitude = mantissa >> shift
    half = 1 << (shift - 1)
    round_up = (remainder > half) | ((remainder == half) & ((magnitude & 1) == 1))
    magnitude = (magnitude + round_up.to(magnitude.dtype)).clamp(max=(1 << magnitude_bits) - 1)
    magnitude = torch.where(exponent == 0, torch.zeros_like(magnitude), magnitude)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float64), (shared - 127 - (magnitude_bits - 1)).to(torch.float64))
    result = magnitude.to(torch.float64) * scale
    result = torch.where((sign == 1) & (magnitude != 0), -result, result)
    return result.to(torch.float32).reshape(values.shape)[..., :width]


def tf32(tensor):
    """A float32 operand as the matrix engine sees it with a 32-bit DEST: the 13 low mantissa bits truncated."""
    return (tensor.to(torch.float32).contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)


def store(tensor, dtype):
    """What a tensor holds after being stored in a device format and read back."""
    if dtype == "float32":
        return tensor.to(torch.float32)
    if dtype == "bfloat16":
        return to_bfloat16(tensor)
    if dtype == "bfloat8_b":
        return bfloat8_roundtrip(tensor)
    if dtype == "bfloat4_b":
        return bfloat8_roundtrip(tensor, magnitude_bits=3)
    raise ValueError(f"no storage model for dtype {dtype}")


def tile_pad(tensor, value=0.0):
    """Pad the last two dimensions up to multiples of 32, as a tiled tensor is padded."""
    height, width = tensor.shape[-2], tensor.shape[-1]
    pad_h = -height % TILE
    pad_w = -width % TILE
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=value)


def to_tiles(tensor):
    """(..., H, W) with H and W multiples of 32 -> (..., H/32, W/32, 32, 32)."""
    *lead, height, width = tensor.shape
    if height % TILE or width % TILE:
        raise ValueError("to_tiles needs tile-aligned H and W; call tile_pad first")
    grid = tensor.reshape(*lead, height // TILE, TILE, width // TILE, TILE)
    return grid.transpose(-3, -2)


def from_tiles(grid):
    """Inverse of to_tiles."""
    *lead, rows, cols, _, _ = grid.shape
    return grid.transpose(-3, -2).reshape(*lead, rows * TILE, cols * TILE)


def randn_offset(shape, offset, generator=None):
    """Data whose mean dwarfs its spread: the format floor and cancellation show up here."""
    return torch.randn(shape, generator=generator) + offset


def with_outliers(shape, factor=1e3, fraction=1e-3, generator=None):
    """Mostly unit-scale data with rare large values: block floats and max-subtraction paths."""
    base = torch.randn(shape, generator=generator)
    mask = torch.rand(shape, generator=generator) < fraction
    return torch.where(mask, base * factor, base)


def _file_hashes(directory):
    """{file name: sha256} of every .py file in `directory`: the oracle files a report belongs to."""
    hashes = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name.endswith(".py") and os.path.isfile(path):
            with open(path, "rb") as handle:
                hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    return hashes


class UpstreamError(RuntimeError):
    """load_upstream() could not read, edit or execute a pinned file, or upstream called into a stand-in."""


class _StandIn:
    """A module that cannot be imported, or a name taken from one (see load_upstream).

    While the pinned text executes it does what import-time code asks of a library name: applied
    as a decorator it returns the function or class as written; called in any other way it
    returns another stand-in (a decorator with arguments, a configuration object); as a base
    class it leaves the class without that base; it survives a subscript and `|` in an
    annotation. Nothing is an instance or a subclass of it: no object of a module that cannot be
    imported exists. Once the load is over, a call raises UpstreamError with the dotted name.
    """

    __slots__ = ("_name", "_state")
    __all__ = ()  # `from stand_in import *` binds nothing

    def __init__(self, name, state):
        self._name, self._state = name, state

    def __getattr__(self, attribute):
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        return _StandIn(f"{self._name}.{attribute}", self._state)

    def __call__(self, *args, **kwargs):
        if not self._state["loading"]:
            raise UpstreamError(f"upstream called {self._name}, which is a stand-in here: the module cannot be imported "
                                f"({self._state['file']}). Replace the call by equivalent torch code with replace=, or take its import out "
                                "with replace= and supply the name with inject= (the report records both), or install the module")
        if len(args) == 1 and not kwargs and isinstance(args[0], (types.FunctionType, type, staticmethod, classmethod, property)):
            return args[0]  # applied as a decorator: the function or class stays as written
        return _StandIn(f"{self._name}(...)", self._state)

    def __mro_entries__(self, bases):
        return ()

    def __getitem__(self, item):
        return _StandIn(f"{self._name}[...]", self._state)

    def __iter__(self):
        raise TypeError(f"{self!r} cannot be iterated")  # without this, __getitem__ would make it an endless sequence

    def __or__(self, other):
        return self

    __ror__ = __or__

    def __instancecheck__(self, instance):
        return False

    def __subclasscheck__(self, subclass):
        return False

    def __repr__(self):
        return f"stand-in for {self._name}"


class _Graft:
    """A package that can be imported, as the pinned text sees it while a stand-in lies below it (`import package.missing`)."""

    __slots__ = ("_module", "_resolve")

    def __init__(self, module, resolve):
        self._module, self._resolve = module, resolve

    def __getattr__(self, attribute):
        found = self._resolve(f"{self._module.__name__}.{attribute}")
        return getattr(self._module, attribute) if found is None else found


def _stand_in_import(missing, state, served):
    """The __import__ a pinned text uses: while it loads, stand-ins for the modules of `missing`; the real import for all else.

    Only the import statements of the text itself come here (the function goes into the builtins of its module, nowhere
    else), so no other module ever meets a stand-in. `served` collects the names of `missing` that the text asked for.
    """
    real_import = builtins.__import__

    def related(name):
        return [stub for stub in missing if name == stub or name.startswith(stub + ".") or stub.startswith(name + ".")]

    def resolve(name):
        """A stand-in for a missing module, an importable package with the stand-ins below it, or None for the real thing."""
        stubs = related(name)
        if not stubs:
            return None
        if all(stub.startswith(name + ".") for stub in stubs) and _importable(name):
            return _Graft(importlib.import_module(name), resolve)
        served.update(stubs)
        return _StandIn(name, state)

    def import_(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.partition(".")[0]
        if level or not state["loading"] or not related(top):
            return real_import(name, globals, locals, fromlist, level)
        served.update(stub for stub in missing if name == stub or name.startswith(stub + "."))
        found = resolve(name)
        if not fromlist:  # `import a.b.c` binds the package a
            if found is None:
                real_import(name, globals, locals, fromlist, level)
            return resolve(top)
        if found is None:
            return real_import(name, globals, locals, fromlist, level)
        if isinstance(found, _Graft):  # `from package import real_name, missing_child`
            real = tuple(entry for entry in fromlist if resolve(f"{name}.{entry}") is None)
            if real:
                real_import(name, globals, locals, real, level)
        return found

    return import_


def _importable(name):
    """Can the module be imported here? Whatever its import raises counts as no."""
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 - a module that fails to import for any reason is not usable here
        return False
    return True


def _syntax_hint(exc, text, compiled_as, edited):
    """The line a SyntaxError of a pinned text points at, quoted from the text as it was compiled.

    Python itself often reads the line of such an error from the file of that name, and that file is the pinned one: after
    a replacement it holds another text, and once a replacement added or removed lines another line stands at that number.
    A syntax error of another source (a module the text imports, a string it compiles) keeps the line Python gives for it.
    """
    line, own = exc.text or "", exc.filename == compiled_as
    if own and exc.lineno is not None:
        lines = re.split(r"\r\n|\r|\n", text)  # the line ends the compiler counts; str.splitlines() knows more
        line = lines[exc.lineno - 1] if 1 <= exc.lineno <= len(lines) else ""
    if not line.strip():
        return ""
    return f": {line.strip()!r}" + (", as the replacements left it" if own and edited else "")


def _bundle_path(path):
    """(the absolute path, the path as the report shows it: relative to the bundle directory where the file lies in it)."""
    absolute = os.path.normpath(path if os.path.isabs(path) else os.path.join(BUNDLE, path))
    relative = os.path.relpath(os.path.realpath(absolute), os.path.realpath(BUNDLE))
    return absolute, (str(path) if relative.startswith("..") else relative).replace(os.sep, "/")


def load_upstream(path, replace=(), stubs=(), inject=None):
    """Execute a pinned upstream file, with literal edits that the report records, and return it as a module.

    For upstream Python that cannot run here as it is pinned: it hard-codes float32 (`.float()`,
    `.to(torch.float32)`, `dtype=torch.float`) where the comparison needs float64, or it imports
    a library that is not installed. The file is only read, never written.

    path     the pinned file: absolute, or relative to the bundle directory (the one above the
             directory of this file), so that it reads as the dossier cites it: sources/...
    replace  (old, new) pairs of literal text, applied in order to the whole text. Each must
             occur at least once in the text it meets, else UpstreamError names it. A relative
             import is yours to replace: ("from ..utils import", "from upstream_parent.utils import")
             with stubs=("upstream_parent",) makes that import a stand-in.
    stubs    names of modules the text imports. Those that cannot be imported here become empty
             stand-in modules, with all below them, for the import statements of this text
             while it loads; no other module ever sees one. `import x`, `from x.y import z` and
             a decorator, a base class or an annotation taken from a stand-in execute, a
             function keeps its body under such a decorator, and nothing is an instance of a
             stand-in. After the load, a call into a stand-in raises UpstreamError with its
             dotted name, and an import of such a module fails as it would without one. A module
             that can be imported is used as it is. Do not list a module whose absence the text
             handles itself (try: import x / except ImportError: ...).
    inject   {name: object}, set in the module before the text runs: a name whose import you
             replaced away, a constant. A name the text binds itself is not yours to inject:
             UpstreamError says so.

    Every load is recorded, and Checks.finish() writes the records into oracle_report.json as
    "upstream_edits": the file, the sha256 of the pinned bytes, each replacement with the number
    of places it changed, the modules that were stand-ins, the injected names. Set nothing on the
    returned module: the report would not show it. Replacing a dtype literal or a missing helper
    library keeps the evidence level of the comparison; editing a statement that computes makes
    the result a transliteration (E3).
    """
    if isinstance(replace, str) or isinstance(stubs, dict) or not isinstance(inject or {}, dict):
        raise UpstreamError("load_upstream(path, replace=[(old, new), ...], stubs=[module name, ...], inject={name: object})")
    replace = list(replace)
    if len(replace) == 2 and all(isinstance(item, str) for item in replace):
        replace = [replace]  # one pair, given bare
    for pair in replace:
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2 and all(isinstance(part, str) for part in pair) and pair[0]):
            raise UpstreamError(f"replace holds {pair!r}: it is a sequence of (old, new) pairs of text, old not empty")
    stubs = [stubs] if isinstance(stubs, str) else list(stubs)
    inject = dict(inject or {})
    for name in stubs:
        if not (isinstance(name, str) and re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", name)):
            raise UpstreamError(f"stubs holds {name!r}: it lists module names, such as einops or fla.utils")
    for name in inject:
        if not (isinstance(name, str) and name.isidentifier()):
            raise UpstreamError(f"inject holds the key {name!r}: it maps plain names of the module to objects")

    absolute, shown = _bundle_path(path)
    try:
        with open(absolute, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise UpstreamError(f"{shown}: cannot read the pinned file ({exc.strerror or exc}); a relative path is taken from {BUNDLE}") from None
    try:
        text = raw.decode(tokenize.detect_encoding(io.BytesIO(raw).readline)[0])
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise UpstreamError(f"{shown}: not Python source text ({exc})") from None
    applied = []
    for position, (old, new) in enumerate(replace, start=1):
        count = text.count(old)
        if not count:
            raise UpstreamError(f"{shown}: replacement {position} of {len(replace)}, {old!r} -> {new!r}, does not occur in the text"
                                + (" that the replacements before it left" if position > 1 else "") + ": a literal must match the pinned file")
        text = text.replace(old, new)
        applied.append([old, new, count])

    state = {"loading": True, "file": shown}
    missing, served = [name for name in stubs if not _importable(name)], set()
    module = types.ModuleType("upstream_%d_%s" % (len(_UPSTREAM_LOADS) + 1, re.sub(r"\W", "_", os.path.splitext(os.path.basename(absolute))[0])))
    module.__file__ = absolute
    module.__dict__.update(inject)
    if missing:  # the import statements of this text, and of no other module, go through the stand-ins
        module.__dict__["__builtins__"] = {**builtins.__dict__, "__import__": _stand_in_import(missing, state, served)}
    sys.modules[module.__name__] = module  # dataclasses and typing look the module of a class up by name
    try:
        exec(compile(text, absolute, "exec", dont_inherit=True), module.__dict__)  # noqa: S102 - pinned reference code
        rebound = sorted(name for name, value in inject.items() if module.__dict__.get(name, value) is not value)
        if rebound:
            raise UpstreamError(f"{shown}: the text binds {', '.join(rebound)} itself (an import, a def or an assignment), so the injected "
                                "object is not the one it uses: take that statement out with replace=, or drop the name from inject=")
    except Exception as exc:  # noqa: BLE001 - whatever the pinned text raises is reported with its file
        sys.modules.pop(module.__name__, None)
        if isinstance(exc, UpstreamError):
            raise
        hint = ""
        if isinstance(exc, SyntaxError):
            hint = _syntax_hint(exc, text, absolute, bool(applied))
        elif isinstance(exc, ImportError) and "relative import" in str(exc):
            hint = (". A relative import is yours to replace: replace=[('from ..utils import', 'from upstream_parent.utils import')] "
                    "with stubs=('upstream_parent',)")
        elif isinstance(exc, ImportError):
            hint = ". List the module in stubs= when the code under test does not need it, or replace its import with replace="
        raise UpstreamError(f"{shown}: the text did not execute: {type(exc).__name__}: {exc}{hint}") from exc
    finally:
        state["loading"] = False

    stood_in = [name for name in missing if name in served]
    record = {"file": shown, "sha256": hashlib.sha256(raw).hexdigest(), "replace": applied, "stubs": stood_in, "inject": sorted(inject)}
    _UPSTREAM_LOADS.append(record)
    places = sum(count for _, _, count in applied)
    detail = "executed as pinned"
    if applied:
        detail = f"{len(applied)} literal replacement{'' if len(applied) == 1 else 's'} in {places} place{'' if places == 1 else 's'}"
    detail += (f"; stand-in modules: {', '.join(stood_in)}" if stood_in else "") + (f"; injected: {', '.join(sorted(inject))}" if inject else "")
    print(f"{'load':5} {'upstream':11} {shown}: {detail}")
    return module


class Checks:
    """Collects named checks, prints one line each, writes oracle_report.json, sets the exit code.

    evidence: the headline, the level reached for the implementation the request is about (the
    code the user's model executes): what the equivalence checks compared reference() with. E1
    that code itself, E2 the upstream repository's pure-Python path, E3 an independent
    transliteration of the kernel or a second implementation, E4 nothing executable to compare
    against. It starts at E4. A comparison with another implementation (another model's Python
    path, a port) states its own level, equal(..., evidence="E3"), and never raises the headline.
    scope: one sentence, which implementation each level covers and what did not run. It starts
    empty and is needed as soon as the passed comparisons carry two or more levels.
    compared_in: the float type both sides of the equivalence checks were computed in: "float64",
    or "float32" for an upstream that computes only in float32.
    finish() adds a FAIL row when these do not fit the checks that ran: a headline of E1, E2 or
    E3 without a passed check of kind "equivalence" at that level; E4 although such a check ran;
    passed comparisons at two or more levels while scope is empty; a level or a float type it
    does not know; an equivalence check that was handed tensors of a float type narrower than
    compared_in.
    That last test reads the dtype of the tensors, and an upstream that computes in float32 and
    casts its result back (.float() ... .type_as(x)) hands over float64. So finish() also looks
    at the size of the largest difference: with compared_in float64, a difference above 1e-12
    and within the gate of 1e-6 is float32-sized, and finish() says so in a line "note: ..."
    and under "notes" in the report. A note never fails the run.
    """

    def __init__(self, op_name):
        self.op_name = op_name
        self.results = []
        self.evidence = "E4"
        self.scope = ""
        self.compared_in = "float64"
        self.worst_equivalence = None

    def equal(self, name, actual, expected, gate=EQUIVALENCE_GATE, kind="equivalence", note="", evidence=None):
        """Math identity: max relative difference, evaluated in float64, at most `gate`.

        kind "equivalence" means reference() against an implementation that ran (upstream()):
        only these checks back an evidence level and set max_rel_diff of the report. Give every
        other comparison its kind: "reformulation" for an F-id of math.md (the name starts with
        the id), "probe" for a known answer or for the oracle against reference().
        evidence: for kind "equivalence" only, the level of this one comparison when it is not
        the headline's: "E1", "E2" or "E3". The row carries it under "evidence"; without it the
        row carries the headline.
        """
        if evidence is not None and kind != "equivalence":
            raise ValueError(f"evidence={evidence!r} belongs to a check of kind equivalence, not {kind}: "
                             "only a comparison with an implementation that ran has a level")
        difference = max_rel_diff(actual, expected)
        passed = difference <= gate
        detail = f"max rel diff {difference:.3e} (gate {gate:.0e})"
        extra = {}
        if kind == "equivalence":
            self.worst_equivalence = max(self.worst_equivalence or 0.0, difference)
            extra["dtypes"] = [str(tensor.dtype).replace("torch.", "") for tensor in (actual, expected)]
            extra["evidence"] = evidence  # None: finish() writes the headline here
            detail += f", level {evidence}" if evidence is not None else ""
        self._add(name, kind, "pass" if passed else "FAIL", detail, note, value=difference, **extra)
        return passed

    def true(self, name, condition, kind="probe", note=""):
        """A structural probe: a boundary, masking or special-value property that must hold."""
        self._add(name, kind, "pass" if condition else "FAIL", "holds" if condition else "VIOLATED", note)
        return bool(condition)

    def precision(self, name, actual, expected, dtype, fp32_dest_acc_en=True, tolerance=None, note=""):
        """Two tensors against a tolerance row, with the suite's metrics.

        Informational: the row gets status "info" and the mark, and never fails the run. Returns
        whether the figure is inside the row. A cell without a default row needs `tolerance`.
        """
        limits = _limits(dtype, fp32_dest_acc_en, tolerance)
        measured = metrics(actual, expected)
        ok = within(measured, dtype, fp32_dest_acc_en, limits)
        detail = f"pcc {measured['pcc']:.6f} rms {measured['rms']:.4f} vs ({limits[0]}, {limits[1]}): {'inside' if ok else 'OUTSIDE'}"
        self._add(name, "precision", "info", f"{detail}; {MARK}", note, value=measured, inside=ok)
        return ok

    def cell(self, name, oracle, inputs, dtype, fp32_dest_acc_en=True, tolerance=None, note="", keep=()):
        """The host check that mirrors what a golden cell of `dtype` charges to the op.

        Informational, like precision(): status "info", the mark, never a failure of the run.
        `oracle(*inputs)` computes the op in float32 from float32 tensors (the pytorch_ function of
        the oracle file does); `inputs` are the float draws a suite would cast to the cell's dtype.
        keep: the positions in `inputs` (0, 1, ...) of tensors that do not take the cell's dtype,
        such as a float32 coordinate tensor in a bfloat16 cell or an index tensor. They reach the
        oracle as they are in every cell, neither cast nor cut nor quantized, and the row names
        them (a tensor that never varies may as well stay inside `oracle`).
        A suite feeds the same stored tensor to reference and device and its reference returns the
        input dtype, so:
        - float32: storage is exact. Reported: oracle(tf32(x)) against oracle(x), what the cell
          loses when the inputs are operands of a matrix-engine phase.
        - bfloat16: input rounding is shared with the reference, so the storage floor is 0.
          Reported: c, the relative RMS of one bfloat16 rounding of the output. A device result
          that lands on the other bfloat16 neighbour for a fraction p of the elements costs
          about sqrt(p) x 3.5 x c: a sensitivity, not a floor.
        - bfloat8_b: the reference sees bfloat16 inputs, the op block-quantized ones, and the
          output is block-quantized again. Reported: that pair, a real floor.
        For a case that draws float32 data and quantizes only on the device, call
        precision(name, store(oracle(store(x, dtype)), dtype), oracle(x), dtype) yourself.
        A cell without a default row needs `tolerance`, else a ValueError says so: bfloat4_b, and
        float32 with fp32_dest_acc_en=False, the cell every op must refuse (its float32 figure
        stays the TF32 cut of a 32-bit DEST).
        """
        inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
        kept = () if keep is None else (keep,) if isinstance(keep, int) else tuple(keep)
        if any(isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(inputs) for position in kept):
            raise ValueError(f"keep={keep!r}: it lists positions in inputs, here 0 to {len(inputs) - 1}")
        kept = sorted(set(kept))
        inputs = tuple(tensor if position in kept else tensor.to(torch.float32) for position, tensor in enumerate(inputs))

        def stored(change):
            """The inputs as the cell holds them: `change` applied to every tensor that is not kept."""
            return tuple(tensor if position in kept else change(tensor) for position, tensor in enumerate(inputs))

        extra = {"keep": kept} if kept else {}
        as_is = ""
        if kept:
            as_is = f"; input{'s' if len(kept) > 1 else ''} {', '.join(map(str, kept))} kept as {'they are' if len(kept) > 1 else 'it is'}"
        limits = _limits(dtype, fp32_dest_acc_en, tolerance)
        if dtype == "float32":
            measured = metrics(oracle(*stored(tf32)), oracle(*inputs))
            meaning = "TF32 cut of the inputs, if they enter a matrix-engine phase"
        elif dtype == "bfloat16":
            output = oracle(*stored(to_bfloat16)).to(torch.float32)
            measured = metrics(to_bfloat16(output), output)
            worst = 3.5 * measured["rms"]
            detail = (f"storage floor 0; one-ULP sensitivity c = {measured['rms']:.4f}: a fraction p of elements on the other "
                      f"neighbour costs sqrt(p) x {worst:.4f} against the rms limit {limits[1]}"
                      + ("; over half the limit at p = 1" if worst > limits[1] / 2 else ""))
            self._add(name, "precision", "info", f"{detail}{as_is}; {MARK}", note,
                      value=measured, sensitivity=worst, inside=worst <= limits[1] / 2, **extra)
            return worst <= limits[1] / 2
        elif dtype in ("bfloat8_b", "bfloat4_b"):
            reference = to_bfloat16(oracle(*stored(to_bfloat16)))
            actual = store(oracle(*stored(lambda tensor: store(to_bfloat16(tensor), dtype))), dtype)
            measured = metrics(actual, reference)
            meaning = "block quantization of inputs and output: a floor"
        else:
            raise ValueError(f"no cell model for dtype {dtype}")
        ok = within(measured, dtype, fp32_dest_acc_en, limits)
        detail = f"{meaning}: pcc {measured['pcc']:.6f} rms {measured['rms']:.4f} vs ({limits[0]}, {limits[1]}): {'inside' if ok else 'OUTSIDE'}"
        self._add(name, "precision", "info", f"{detail}{as_is}; {MARK}", note, value=measured, inside=ok, **extra)
        return ok

    def skipped(self, name, reason, kind="equivalence"):
        self.results.append({"name": name, "kind": kind, "status": "skipped", "detail": reason})
        print(f"skip  {kind:11} {name}: {reason}")

    def _add(self, name, kind, status, detail, note, **extra):
        self.results.append({"name": name, "kind": kind, "status": status, "detail": detail, "note": note, **extra})
        print(f"{status:5} {kind:11} {name}: {detail}" + (f"  ({note})" if note else ""))

    def _check_evidence(self):
        """FAIL rows for evidence levels or a float type that the comparisons do not back; returns the levels of the passed comparisons.

        Every equivalence check that ran gets its level here: the one it was given, else the headline.
        """
        ran = [row for row in self.results if row["kind"] == "equivalence" and row["status"] in ("pass", "FAIL")]
        unknown = [row for row in ran if row.get("evidence") is not None and row["evidence"] not in COMPARISON_LEVELS]
        for row in ran:
            if row.get("evidence") is None:
                row["evidence"] = self.evidence
        passed = [row for row in ran if row["status"] == "pass" and "value" in row]  # comparisons made by equal()
        levels = sorted({str(row["evidence"]) for row in passed})
        if self.evidence not in EVIDENCE_LEVELS:
            self._add("evidence level", "evidence", "FAIL", f"evidence is {self.evidence!r}: it is one of {', '.join(EVIDENCE_LEVELS)}", "")
        elif self.evidence != "E4" and not passed:
            self._add("evidence level", "evidence", "FAIL", f"evidence {self.evidence} needs an executed comparison: "
                      "no check of kind equivalence passed (checks.equal against upstream)", "")
        elif self.evidence != "E4" and self.evidence not in levels:
            self._add("evidence level", "evidence", "FAIL",
                      f"evidence {self.evidence} is the headline, but no check of kind equivalence passed at that level "
                      f"(passed: {', '.join(levels)}). The headline is the level reached for the implementation the request is about: set it "
                      "to what was compared with that implementation, or run that comparison; another implementation keeps its level on "
                      "its own rows", "")
        elif self.evidence == "E4" and ran:
            self._add("evidence level", "evidence", "FAIL",
                      f"evidence says E4 (nothing executed), but checks of kind equivalence ran ({len(ran)}): set it to what they compared "
                      'with (E1, E2 or E3); a comparison with no implementation behind it takes kind="probe"', "")
        if unknown:
            found = sorted({repr(row["evidence"]) for row in unknown})
            self._add("evidence of a comparison", "evidence", "FAIL",
                      f"equivalence checks carry the level {', '.join(found)} ({len(unknown)}, first: {unknown[0]['name']}): "
                      f"a comparison that ran is at {', '.join(COMPARISON_LEVELS[:2])} or {COMPARISON_LEVELS[2]}", "")
        if len(levels) > 1 and not " ".join(str(self.scope or "").split()):
            self._add("evidence scope", "evidence", "FAIL",
                      f"the passed comparisons are at {' and '.join(levels)}, and checks.scope is empty: say in one sentence which "
                      "implementation each level covers and what did not run", "")
        self._check_compared_in(ran)
        return levels

    def _check_compared_in(self, ran):
        """FAIL rows for a float type that the equivalence checks which ran do not back."""
        if self.compared_in not in NARROWER:
            self._add("compared_in", "evidence", "FAIL",
                      f"compared_in is {self.compared_in!r}: it is float64, or float32 for an upstream that computes only in float32", "")
            return
        narrow = [row for row in ran if any(dtype in NARROWER[self.compared_in] for dtype in row.get("dtypes", ()))]
        if narrow:
            found = sorted({dtype for row in narrow for dtype in row["dtypes"] if dtype in NARROWER[self.compared_in]})
            advice = ('compute both sides in float64, or set checks.compared_in = "float32" for an upstream that computes only in float32'
                      if self.compared_in == "float64" and found == ["float32"] else
                      "an equivalence check runs in float64, or in float32 for a float32-only upstream, never in a 16-bit type")
            self._add("compared_in", "evidence", "FAIL",
                      f"compared_in says {self.compared_in}, but equivalence checks were handed {', '.join(found)} tensors "
                      f"({len(narrow)}, first: {narrow[0]['name']}): {advice}", "")

    def _notes(self):
        """Remarks that never fail a run: finish() prints each as a line "note: ..." and writes them under "notes".

        The one there is: compared_in is float64 and the largest difference of the equivalence checks is float32-sized.
        """
        notes = []
        if self.compared_in == "float64" and self.worst_equivalence is not None and FLOAT64_SIZED < self.worst_equivalence <= EQUIVALENCE_GATE:
            notes.append(FLOAT32_SIZED_NOTE)
        return notes

    def finish(self, directory=None):
        """Write oracle_report.json beside the script (or into `directory`), print the summary, exit.

        The evidence levels, the scope and compared_in are first held against the comparisons that
        ran (class docstring), and every load_upstream() of the run goes into "upstream_edits".
        A float32-sized largest difference of a float64 comparison goes into "notes" and is
        printed as a line "note: ..." below the summary: it changes no row and no exit code.
        Exit code 1 when a row has status FAIL, else 0; a precision row never has.
        """
        levels = self._check_evidence()
        notes = self._notes()
        scope = " ".join(str(self.scope or "").split())
        failed = [result for result in self.results if result["status"] == "FAIL"]
        folder = directory or os.path.dirname(os.path.abspath(sys.argv[0]))
        target = os.path.join(folder, "oracle_report.json")
        try:
            report = {
                "op": self.op_name,
                "evidence": self.evidence,
                "evidence_scope": scope,
                "evidence_levels": levels,
                "max_rel_diff": self.worst_equivalence,
                "compared_in": self.compared_in,
                "passed": not failed,
                "notes": notes,
                "torch": torch.__version__,
                "python": sys.version.split()[0],
                "date": datetime.date.today().isoformat(),
                "files": _file_hashes(folder),
                "upstream_edits": list(_UPSTREAM_LOADS),
                "checks": self.results,
            }
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=1, default=str)
                handle.write("\n")
        except OSError as exc:
            print(f"error: cannot write {target}: {exc}")
            sys.exit(2)
        counted = [result for result in self.results if result["status"] in ("pass", "FAIL")]
        informational = [result for result in self.results if result["status"] == "info"]
        skipped = [result for result in self.results if result["status"] == "skipped"]
        summary = f"{self.op_name}: evidence {self.evidence}"
        others = [level for level in levels if level != self.evidence]
        if others:
            summary += f" with other comparisons at {', '.join(others)}"
        if self.worst_equivalence is not None:
            summary += f" (largest relative difference {self.worst_equivalence:.1e}, compared in {self.compared_in})"
        summary += f", {len(counted) - len(failed)}/{len(counted)} checks pass"
        if skipped:
            summary += f", {len(skipped)} skipped"
        if informational:
            flagged = sum(1 for result in informational if not result.get("inside", True))
            summary += f"; {len(informational)} precision rows are informational ({flagged} flagged: OUTSIDE, or over half the limit)"
        print(f"{summary}; report in {target}")
        if scope:
            print(f"scope: {scope}")
        for note in notes:
            print(f"note: {note}")
        sys.exit(1 if failed else 0)
