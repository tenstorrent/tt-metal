# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Inspect upstream dispatch and verify the calls that actually reach JIT.

Names bind public APIs, never numerical implementations. Unknown dispatch
shapes are reported, not silently mistaken for parameterless unary calls.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re


DISPATCH = Path("ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp")


def _balanced_end(code: str, opening: int, left: str, right: str) -> int:
    depth = 1
    end = opening + 1
    while end < len(code) and depth:
        depth += (code[end] == left) - (code[end] == right)
        end += 1
    if depth:
        raise ValueError(f"unbalanced upstream {left}{right} at {opening}")
    return end


def _signature_parameters(signature: str) -> list[str]:
    """Split declarations, keeping template arguments and initializers intact."""
    stack = []
    parts = []
    start = 0
    pairs = {">": "<", ")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(_code(signature)):
        if char in "<([{":
            stack.append(char)
        elif char in pairs and stack and stack[-1] == pairs[char]:
            stack.pop()
        elif char == "," and not stack:
            parts.append(" ".join(signature[start:index].split()))
            start = index + 1
    parts.append(" ".join(signature[start:].split()))
    return [part for part in parts if part]


def _public_definitions(root: Path) -> dict[str, tuple[dict[str, str], ...]]:
    """Index actual eltwise entry definitions, not a maintained op-name list.

    This is deliberately an inventory, not a C++ parser or proof of packaging
    support. Unknown signatures and composite bodies remain visible liabilities.
    Read the current checkout each time: packaging edits dispatch in place, so
    caching by checkout path would silently return pre-edit API facts.
    """
    directory = root / "ttnn/cpp/ttnn/operations/eltwise"
    result: dict[str, list[dict[str, str]]] = {}
    for path in sorted(directory.rglob("*.cpp")):
        if "kernels" in path.relative_to(directory).parts or path.name.endswith("_device_operation.cpp"):
            continue
        text = path.read_text()
        code = _code(text)
        # Expand upstream's own API registration macros, keeping their formal
        # signature/body. Macro identity does not select a numerical evaluator.
        macros = {}
        for macro in re.finditer(r"(?m)^#define\s+(DEFINE_UNARY_OP\w*)\([^\n]*", text):
            stop = macro.end()
            while text[stop - 1 : stop] == "\\":
                stop = text.find("\n", stop + 1)
                if stop < 0:
                    stop = len(text)
                    break
            macros[macro.group(1)] = text[macro.end() : stop].replace("\\\n", "\n")
        expanded = []
        for match in re.finditer(r"(?m)^(DEFINE_UNARY_OP\w*)\(\s*(\w+)\s*,\s*(\w+)\s*\)", code):
            template = macros.get(match.group(1))
            if template:
                expanded.append(template.replace("op_name", match.group(2)).replace("OP_TYPE", match.group(3)))
        code += "\n" + _code("\n".join(expanded))
        for match in re.finditer(r"(?m)^\s*(?:Tensor|std::vector<[^\n]+>)\s+(\w+)\s*\(", code):
            opening = code.find("(", match.start())
            signature_end = _balanced_end(code, opening, "(", ")")
            body_start = signature_end
            while body_start < len(code) and code[body_start].isspace():
                body_start += 1
            if code[body_start : body_start + 1] != "{":
                continue
            body_end = _balanced_end(code, body_start, "{", "}")
            signature = code[opening + 1 : signature_end - 1]
            body = code[body_start:body_end]
            parameters = _signature_parameters(signature)
            tensors = [p for p in parameters if re.search(r"\b(?:const\s+)?Tensor\s*&", p)]
            values = [p for p in parameters if not re.search(r"Tensor|MemoryConfig|CoreRangeSet|QueueId", p)]
            symbols = sorted(set(re.findall(r"UnaryOpType::(\w+)", body)))
            item = {
                "tensor_arity": str(len(tensors)),
                "value_parameters": "; ".join(values),
                "symbols": ",".join(symbols),
                "implementation": "unary_dispatch" if "unary_impl(" in body and symbols else "composite_or_custom",
                "definition": str(path.relative_to(root)),
            }
            result.setdefault(match.group(1), []).append(item)
    return {name: tuple(items) for name, items in result.items()}


def inspect_public_api(root: Path, operation: str) -> dict[str, str]:
    """Describe public overloads without claiming their LLK backend exists."""
    definitions = _public_definitions(root.resolve()).get(operation, ())
    return {
        "public_api": operation,
        "api_status": "found" if definitions else "not_found",
        "tensor_arity": ",".join(sorted({d["tensor_arity"] for d in definitions})),
        "value_parameters": " | ".join(dict.fromkeys(d["value_parameters"] for d in definitions)),
        "symbols": ",".join(sorted({s for d in definitions for s in d["symbols"].split(",") if s})),
        "implementation": ",".join(sorted({d["implementation"] for d in definitions})),
        "definitions": ",".join(sorted({d["definition"] for d in definitions})),
    }


def _code(text: str) -> str:
    """Mask comments and quoted literals without moving source offsets."""
    return re.sub(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        lambda match: re.sub(r"[^\n]", " ", match.group()),
        text,
    )


@dataclass(frozen=True)
class DispatchArm:
    scope: str
    symbol: str
    start: int
    end: int
    body: str


def dispatch_arms(source: str, operation: str) -> tuple[DispatchArm, ...]:
    """Enumerate every arm in both upstream unary dispatch functions."""
    code = _code(source)
    arms = []
    for function in re.finditer(r"\bget_op_init_and_func_(default|parameterized)\s*\(", code):
        opening = code.find("{", function.end())
        if opening < 0:
            raise ValueError("dispatch function has no body")
        depth = 1
        end = opening + 1
        while end < len(code) and depth:
            depth += (code[end] == "{") - (code[end] == "}")
            end += 1
        if depth:
            raise ValueError("unbalanced dispatch function")
        region = code[opening + 1 : end - 1]
        labels = list(re.finditer(r"\bcase\s+UnaryOpType::(\w+)\s*:|\bdefault\s*:", region))
        # Only labels at the outer switch depth; nested enum switches are not
        # independent public dispatch paths.
        labels = [m for m in labels if region[: m.start()].count("{") - region[: m.start()].count("}") == 1]
        for index, label in enumerate(labels):
            if label.group(1) != operation.upper():
                continue
            start = opening + 1 + label.end()
            stop = opening + 1 + (labels[index + 1].start() if index + 1 < len(labels) else len(region))
            body = source[start:stop].rstrip()
            # The final arm is bounded by the switch's closing brace.
            if index + 1 == len(labels):
                masked = _code(body)
                nesting = 0
                for pos, token in enumerate(masked):
                    if token == "}" and nesting == 0:
                        body = body[:pos].rstrip()
                        break
                    nesting += (token == "{") - (token == "}")
            arms.append(DispatchArm(function.group(1), operation.upper(), start, start + len(body), body))
    return tuple(arms)


def arm_kind(arm: DispatchArm, operation: str) -> str:
    compact = re.sub(r"\s+", "", arm.body)
    op = re.escape(operation)
    if re.fullmatch(rf'return\{{"{op}_tile_init\(\);",fmt::format\("{op}_tile\(\{{0?\}}\);",idst\)\}};', compact):
        return "default"
    if re.fullmatch(
        rf'return\{{fmt::format\("{op}_tile_init<\{{\}}u>\(\);",\(uint32_t\)param0\),fmt::format\("{op}_tile<\{{1\}}u>\(\{{0\}}\);",idst,\(uint32_t\)param0\)\}};',
        compact,
    ):
        return "template_mode"
    return "unmodelled"


def inspect_surface(root: Path, operation: str) -> dict[str, str]:
    source = (root / DISPATCH).read_text()
    public = inspect_public_api(root, operation)
    symbols = (
        public["symbols"].split(",")
        if public["symbols"] and public["implementation"] == "unary_dispatch"
        else [operation.upper()]
    )
    arms = tuple(arm for symbol in symbols for arm in dispatch_arms(source, symbol))
    registration = root / "ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp"
    registered = _code(registration.read_text()) if registration.exists() else ""
    fast_mode = bool(
        re.search(
            rf"(?m)^DEFINE_UNARY_OP_WITH_FAST_AND_APPROXIMATE_MODE\(\s*{re.escape(operation)}\s*,\s*{re.escape(operation.upper())}\s*\)",
            registered,
        )
    )
    return {
        "operation": operation,
        "entry": "fast_approximate_mode" if fast_mode else "not_inferred",
        "arms": ",".join(f"{arm.scope}:{arm_kind(arm, operation)}" for arm in arms)
        or ("composite_or_custom_api" if public["api_status"] == "found" else "public_api_not_found"),
    }


def _without_comments(text: str) -> str:
    """Keep quoted dispatch arguments; comments cannot provide JIT evidence."""
    return re.sub(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        lambda match: re.sub(r"[^\n]", " ", match.group()) if match.group().startswith(("//", "/*")) else match.group(),
        text,
    )


def verify_public_source_closure(root: Path, sources) -> None:
    """Compare canonical compiler-owned source bytes, without a hash registry."""
    for relative, expected in sources:
        if (root / relative).read_bytes() != expected:
            raise ValueError("public candidate has different shared source: " + relative)


def verify_public_stock_dispatch(cache: Path) -> None:
    """A stock comparison must not execute a generated public dispatch."""
    if any("TT_POLY_SELECTED_CONFIG_HEADER" in path.read_text() for path in cache.rglob("defines_generated.h")):
        raise ValueError("stock arm contains generated dispatch")


def verify_public_jit_selection(
    cache: Path, expected_cpp: str, expected_defines: dict[str, str], *, root: Path | None = None
) -> int:
    """Verify every public-unary JIT instance against its typed compiler contract.

    ``expected_defines`` comes from ``lifecycle_defines(program)``. Aliases may
    select a different config filename, but its bytes must match exactly.
    Tensor-transfer utility kernels are not unary dispatch; stock or conflicting
    entries under the actual ``eltwise_sfpu`` compute shell fail this check.
    A dedicated per-operation cache is required, as in the public batch runner.
    This proves JIT selection, not successful execution or numerical accuracy.
    Package-relative config paths resolve only against the explicitly supplied
    TT-Metal checkout, never the current working directory or an inferred root.
    """
    expected = {key: str(value).strip() for key, value in expected_defines.items()}
    if expected.get("SFPU_OP_CHAIN_0") != "ttpoly_compiled_tile();" or not expected.get("SFPU_OP_PROGRAM_INIT_0"):
        raise ValueError("public JIT check requires the typed shared lifecycle contract")
    expected.pop("TT_POLY_SELECTED_CONFIG_HEADER", None)
    # New TT-Metal dispatch firmware can run on TRISC too. Its source directory
    # defines that infrastructure role; it is not an activation-name exception.
    dispatch_sources = (root / "tt_metal/impl/dispatch/kernels") if root is not None else None
    dispatch_names = (
        {path.stem for path in dispatch_sources.rglob("*.cpp")}
        if dispatch_sources is not None and dispatch_sources.is_dir()
        else set()
    )
    observed = 0
    for header in sorted(cache.rglob("defines_generated.h")):
        text = _without_comments(header.read_text().replace("\\\n", ""))
        selected_marker = re.search(r"(?m)^\s*#\s*define\s+TT_POLY_SELECTED_CONFIG_HEADER\b", text)
        infrastructure = header.parent.parent.name in dispatch_names
        if infrastructure and not re.search(r"(?m)^\s*#\s*define\s+(?:SFPU_OP_|TT_POLY_)", text):
            continue
        unary_shell = "eltwise_sfpu" in header.relative_to(cache).parts
        # DeviceOperation shells need not be named eltwise_sfpu. Actual math
        # JITs carry the TRISC1 artifact directory; data-movement/dispatch
        # helpers do not. Reject extra stock math in these per-call caches.
        math_shell = (header.parent / "trisc1").is_dir()
        flat_fixture = header.parent == cache and re.search(r"(?m)^\s*#\s*define\s+SFPU_OP_CHAIN_", text)
        if not (selected_marker or unary_shell or math_shell or flat_fixture):
            continue
        if re.search(r"(?m)^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|undef)\b", text):
            raise ValueError(f"public JIT has conditional or undefined selection: {header}")
        definitions = {}
        for line in text.splitlines():
            if not re.match(r"\s*#\s*define\b", line):
                continue
            match = re.fullmatch(r"\s*#\s*define\s+([A-Za-z_]\w*)(?:\s+(.*?))?\s*", line)
            if not match or match[1] in definitions:
                raise ValueError(f"public JIT has malformed or duplicate definitions: {header}")
            definitions[match[1]] = (match[2] or "").strip()
        selected = definitions.get("TT_POLY_SELECTED_CONFIG_HEADER", "")
        path_match = re.fullmatch(r'"([^"\n]+)"', selected)
        if not path_match:
            raise ValueError(f"public JIT selected stock or has no exact config path: {header}")
        config = Path(path_match[1])
        if ".." in config.parts:
            raise ValueError(f"public JIT config path contains traversal: {header}")
        if not config.is_absolute():
            if root is None:
                raise ValueError(f"public JIT config path must resolve explicitly: {header}")
            config = root.resolve() / config
        if root is not None and not config.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"public JIT config escapes its explicit checkout: {header}")
        try:
            actual_cpp = config.read_bytes()
        except OSError as error:
            raise ValueError(f"public JIT config is unavailable: {config}") from error
        if actual_cpp != expected_cpp.encode():
            raise ValueError(f"public JIT selected a different canonical program: {header}")
        mismatched = [key for key, value in expected.items() if definitions.get(key) != value]
        unexpected = [
            key
            for key in definitions
            if key.startswith(("SFPU_OP_", "TT_POLY_"))
            and key not in expected
            and key != "TT_POLY_SELECTED_CONFIG_HEADER"
        ]
        if mismatched or unexpected:
            raise ValueError(
                f"public JIT lifecycle mismatch at {header}: changed={mismatched}, unexpected={unexpected}"
            )
        observed += 1
    if not observed:
        raise ValueError(f"public JIT did not select the canonical lifecycle in {cache}")
    return observed


def verify_jit_selection(overlay: Path, cache: Path, operation: str) -> int:
    """Require exact generated calls, not helper names mentioned in a cache.

    The supported unary surface is one tile call per FUNC macro. Additional
    calls, expressions, conditional definitions, or unknown argument forms
    require an explicit verifier extension rather than substring acceptance.
    """
    if not re.fullmatch(r"[A-Za-z_]\w*", operation):
        raise ValueError(f"invalid public operation identifier: {operation!r}")

    source = _without_comments((overlay / DISPATCH).read_text())
    op = re.escape(operation)
    pattern = re.compile(
        rf"(?<![\w:])({op}_tt_poly_bf16_tile|{op}_tile)" r"\s*(<\s*(?:true|false)\s*>)?\s*\(([^()]*)\)"
    )

    def uint32_word(token: str) -> int:
        # Do not interpret a leading-zero C++ octal word as Python decimal.
        if not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|0|[1-9][0-9]*)[uU]?", token):
            raise ValueError("expected a literal uint32 argument")
        spelling = token.rstrip("uU")
        value = int(spelling, 16 if spelling.lower().startswith("0x") else 10)
        if value > 0xFFFFFFFF:
            raise ValueError("uint32 argument is out of range")
        return value

    expected = set()
    for match in pattern.finditer(source):
        mode = re.sub(r"\s+", "", match[2] or "")
        if not match[1].endswith("_tt_poly_bf16_tile") and mode != "<true>":
            continue
        arguments = [argument.strip() for argument in match[3].split(",")]
        try:
            if arguments[0] not in ("{}", "{0}"):
                uint32_word(arguments[0])
            scalar_words = tuple(uint32_word(argument) for argument in arguments[1:])
        except ValueError as error:
            raise ValueError("overlay must declare literal scalar words after its tile index") from error
        expected.add((match[1], mode, scalar_words))
    if len({name for name, _, _ in expected}) != 1:
        raise ValueError("overlay must declare exactly one generated call surface")
    selected = ", ".join(
        name + mode + "(idst" + "".join(f", 0x{word:08x}u" for word in words) + ")"
        for name, mode, words in sorted(expected)
    )
    headers = sorted(cache.rglob("defines_generated.h"))
    observed = 0
    call = re.compile(r"([A-Za-z_]\w*)\s*(<\s*(?:true|false)\s*>)?" r"\s*\(([^()]*)\)\s*;")
    for header in headers:
        text = _without_comments(header.read_text().replace("\\\n", ""))
        definitions = set()
        for line_number, line in enumerate(text.splitlines(), 1):
            if not re.match(r"\s*#\s*define\s+SFPU_OP_CHAIN_\w*_FUNC_", line):
                continue
            match = re.fullmatch(
                r"\s*#\s*define\s+(SFPU_OP_CHAIN_\d+_FUNC_\d+)\s+(.+?)\s*",
                line,
            )
            invocation = call.fullmatch(match[2]) if match else None
            signature = None
            if invocation:
                try:
                    arguments = tuple(uint32_word(argument.strip()) for argument in invocation[3].split(","))
                    signature = (invocation[1], re.sub(r"\s+", "", invocation[2] or ""), arguments[1:])
                except ValueError:
                    pass
            if (
                not match
                or not invocation
                or signature not in expected
                or match[1] in definitions
                or re.search(r"(?m)^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b", text)
            ):
                raise ValueError(
                    f"JIT did not exclusively select {selected}: " f"{header}:{line_number}: {line.strip()}"
                )
            definitions.add(match[1])
            observed += 1
    if not observed:
        raise ValueError(f"JIT did not select {selected}: no SFPU FUNC call evidence in {cache}")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    census = commands.add_parser("census")
    census.add_argument("--tt-metal-root", type=Path, required=True)
    census.add_argument("--queue", type=Path, required=True)
    census.add_argument(
        "--spec-root", type=Path, help="activation JSON directory; resolves declared public op_name aliases"
    )
    check = commands.add_parser("check-jit")
    check.add_argument("--overlay", type=Path, required=True)
    check.add_argument("--cache", type=Path, required=True)
    check.add_argument("--operation", required=True)
    args = parser.parse_args()
    if args.command == "check-jit":
        count = verify_jit_selection(args.overlay, args.cache, args.operation)
        print(f"JIT dispatch: {count} generated call(s), no stock fallback")
    else:
        lines = args.queue.read_text().splitlines()
        column = lines[0].split("\t").index("activation")
        print(
            "operation\tentry\tdispatch_arms\tpublic_api\tapi_status\ttensor_arity\tvalue_parameters\tsymbols\timplementation\tdefinitions"
        )
        for line in lines[1:]:
            if line:
                activation = line.split("\t")[column]
                operation = activation
                if args.spec_root:
                    spec = json.loads((args.spec_root / f"{activation}.json").read_text())
                    operation = spec.get("op_name", activation)
                row = inspect_surface(args.tt_metal_root, operation)
                row["operation"] = activation
                print("\t".join((*row.values(), *inspect_public_api(args.tt_metal_root, operation).values())))


if __name__ == "__main__":
    main()
