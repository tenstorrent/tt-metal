# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused greedy tail without a device: its registry entry, the three kernels' argument contracts against the Python
side, the bf16 / fp32 sign-magnitude order keys against torch's order, the staging layouts against the CBs, the composed
chains as subsequences of the LM head's own op sequences, and the LM-head hook and dataclass field pinned in
``embedding.py``."""

from __future__ import annotations

import ast
import inspect
import json
import re
import struct
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import greedy_tail as gt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

HERE = Path(__file__).resolve().parents[1]
EMBEDDING_SOURCE = (HERE / "ttnn" / "embedding.py").read_text()
SOURCES = {name: (fp.REPO_ROOT / path).read_text() for name, path in gt.KERNELS.items()}


def _ttnn_calls(source: str) -> list[str]:
    tree = ast.parse(inspect.cleandoc(source) if not source.startswith("def") else source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name) and value.id == "ttnn":
                calls.append((node.lineno, node.col_offset, node.func.attr))
    return [name for _, _, name in sorted(calls)]


def _method_source(class_name: str, method: str) -> str:
    body = EMBEDDING_SOURCE[EMBEDDING_SOURCE.index(f"\nclass {class_name}") :]
    match = re.search(rf"\n    def {method}\(.*?(?=\n    def |\n\S)", body, re.S)
    assert match, (class_name, method)
    return inspect.cleandoc(match.group(0))


def _is_subsequence(short: list[str], long: list[str]) -> bool:
    it = iter(long)
    return all(any(item == other for other in it) for item in short)


def test_registered_bitwise():
    entry = fused.kernel("greedy_tail")
    assert entry.tolerance == fused.BITWISE and entry.gate is None
    assert entry.fused is gt.greedy_candidates_fused and entry.composed is gt.greedy_candidates_chain
    default = gt.greedy_candidates_fused if "greedy_tail" in fused.DEFAULT_ON else gt.greedy_candidates_chain
    assert fused.resolve("greedy_tail", {}) is default  # the registry list decides
    assert fused.resolve("greedy_tail", {fused.OFF_ENV: "greedy_tail"}) is gt.greedy_candidates_chain
    assert fused.resolve("greedy_tail", {fused.ENV: "greedy_tail"}) is gt.greedy_candidates_fused
    fused_params = dict(inspect.signature(gt.greedy_candidates_fused).parameters)
    assert fused_params.pop("candidate_row").default is None  # the fold's input; the chain has no candidate row
    assert fused_params.keys() == inspect.signature(gt.greedy_candidates_chain).parameters.keys()


@pytest.mark.parametrize(
    "kernel, runtime_args, tensors, named",
    [
        ("scan", gt.SCAN_ARGS, 3, {"cb_stage", "lanes_per_tile", "rows", "candidates", "lane_split"}),
        ("merge", gt.MERGE_ARGS, 8, {"cb_stage", "cores", "rows", "packed_lanes", "candidates"}),
        ("resolve", gt.RESOLVE_ARGS, 6, {"cb_stage", "devices", "copy_into", "rows", "stride"}),
    ],
)
def test_kernel_arg_contracts(kernel, runtime_args, tensors, named):
    source = SOURCES[kernel]
    used = sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", source)})
    assert used == list(range(len(runtime_args)))  # the candidate-row tensors' args sit last (read with candidates > 0)
    assert "TensorAccessorArgs<0>()" in source and source.count("next_compile_time_args_offset()") == tensors - 1
    assert set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', source)) == named
    assert "void kernel_main()" in source


def test_python_side_named_args_match_the_kernels():
    source = inspect.getsource(gt)
    for kernel, expected in (
        ("scan", {"cb_stage", "lanes_per_tile", "rows", "candidates", "lane_split"}),
        ("merge", {"cb_stage", "cores", "rows", "packed_lanes", "candidates"}),
        ("resolve", {"cb_stage", "devices", "copy_into", "rows", "stride"}),
    ):
        block = source.split(f'KERNELS["{kernel}"]')[1].split("named={")[1].split("}")[0]
        assert set(re.findall(r'"([a-z_0-9]+)":', block)) == expected, kernel


def _key16(bits: int) -> int:
    return (~bits & 0xFFFF) if bits & 0x8000 else (bits | 0x8000)


def _key32(bits: int) -> int:
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


def test_sign_magnitude_keys_order_as_the_floats():
    assert "return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);" in SOURCES["scan"]
    assert "return (bits & 0x7FFFu) ? bits : uint16_t(0);" in SOURCES["scan"]  # -0.0 -> +0.0 before keying
    assert "const uint16_t bits = canonical(halves[word]);" in SOURCES["scan"]
    assert "fp32_bits = (fp32_bits & 0x7FFFFFFFu) ? fp32_bits : 0u;" in SOURCES["merge"]
    assert "return (fp32_bits & 0x80000000u) ? ~fp32_bits : (fp32_bits | 0x80000000u);" in SOURCES["merge"]
    g = torch.Generator().manual_seed(1)
    values = torch.cat([torch.randn(4096, generator=g) * 40, torch.tensor([-1e30, 1e30, -0.0, 0.0, 1e-38, -1e-38])])
    bf16 = values.to(torch.bfloat16)
    order = torch.argsort(bf16.float(), stable=True)
    floats = bf16[order].float().tolist()
    keys = [_key16(0 if (int(b) & 0x7FFF) == 0 else int(b) & 0xFFFF) for b in bf16[order].view(torch.int16)]
    for (k0, f0), (k1, f1) in zip(zip(keys, floats), zip(keys[1:], floats[1:])):
        assert (k0 < k1) if f0 < f1 else (k0 == k1)  # zeros canonicalized: equal floats have equal keys
    keys32 = [_key32(struct.unpack("<I", struct.pack("<f", f))[0]) for f in floats]
    assert all(a <= b for a, b in zip(keys32, keys32[1:]))


def test_scan_visits_lanes_in_id_order_and_keeps_the_first_maximum():
    source = SOURCES["scan"]
    assert "if (!any || key > best_key)" in source  # strict: an equal later lane never replaces the lower id
    assert "best_id = (first + t) * LANES + lane;" in source
    assert "(lane < 16 ? lane : 32 + (lane - 16))" in source  # face 0 row 0 then face 1 row 0 of the 128-byte stage
    assert "if (c == 0 || key > best_key)" in SOURCES["merge"]
    assert "id_fp32.f = static_cast<float>(best_id);" in SOURCES["merge"]  # the packed lane is float(id), not its bits
    assert (
        "words[STAGE_OUT / 4 + 1] = id_fp32.u;" in SOURCES["merge"]
        and "words[STAGE_OUT / 4 + 4] = best_id;" in SOURCES["merge"]
    )
    assert "if (ranked > best)" in SOURCES["resolve"] and "for (uint32_t d = 1; d < DEVICES; ++d)" in SOURCES["resolve"]


def test_staging_layouts_fit_the_cbs():
    tiles = 62080 // fp.TILE
    per_core = -(-tiles // gt.SCAN_CORES)
    assert gt.scan_cores({}) == gt.SCAN_CORES == 40 and gt.scan_cores({gt.SCAN_CORES_ENV: "80"}) == 80
    assert 128 * per_core + 16 <= gt.SCAN_STAGE_PAGES * 2048
    pairs_bytes = ((16 * gt.SCAN_CORES) + 63) & ~63
    assert 2048 + pairs_bytes + 32 <= gt.MERGE_STAGE_PAGES * 2048
    assert 4096 + 3 * 64 <= gt.RESOLVE_STAGE_PAGES * 4096
    assert "PAIRS_BYTES = ((16 * CORES) + 63) & ~63u;" in SOURCES["merge"]
    assert "pairs_lanes = -(-4 * pairs_per_row // 16) * 16" in inspect.getsource(gt.greedy_candidates)
    assert (
        "{.page_id = row, .offset_bytes = 16 * core}" in SOURCES["scan"]
    )  # row 0 of the one-row step, row r of a lane


def test_composed_chains_follow_the_lm_head_op_order():
    candidates = _ttnn_calls(inspect.getsource(gt.greedy_candidates_composed))
    chain = _ttnn_calls(_method_source("Qwen38TTNNLMHead", "greedy_candidates"))
    core = [c for c in candidates if c not in ("deallocate", "typecast", "concat")][:7]
    assert core == ["to_layout", "argmax", "pad", "reshape", "max", "reshape", "max"]
    assert _is_subsequence(core, chain)
    resolve = [c for c in _ttnn_calls(inspect.getsource(gt.resolve_composed)) if c != "deallocate"]
    assert resolve == [
        "to_layout",
        "typecast",
        "subtract",
        "argmax",
        "typecast",
        "add",
        "gather",
        "multiply",
        "to_layout",
    ]
    assert _is_subsequence(resolve, _ttnn_calls(_method_source("Qwen38TTNNLMHead", "resolve_greedy_on_device")))


def test_fused_resolve_gathers_once_along_the_tp_axis():
    source = inspect.getsource(gt.resolve_greedy_on_device_fused)
    assert source.count("ttnn.all_gather(") == 1 and "dim=3, cluster_axis=embedding.TP_AXIS" in source
    assert 'raise RuntimeError("on-device greedy resolve requires Linear topology")' in source
    assert "placement=TensorPlacement.LOCAL_PARTIAL" in source and "placement=TensorPlacement.REPLICATED" in source
    assert "return type(lm_head).resolve_greedy_on_device(lm_head, candidates, into=into)" in source  # chain candidates


def test_token_copy_is_the_resolves_second_write():
    """The server's ttnn.copy(token_row, token_row_io) after the resolve is the resolve program's second 4 KB write
    (``into``); the chain method copies after its own resolve; the MTP body keeps its copy after the MTP row."""

    assert "if constexpr (COPY_INTO) {" in SOURCES["resolve"] and SOURCES["resolve"].count("noc.async_write(") == 2
    assert gt.RESOLVE_ARGS[-1] == "into_addr" and len(gt.RESOLVE_ARGS) == 6
    source = inspect.getsource(gt.resolve)
    assert "token_row if into is None else into]" in source and '"copy_into": 0 if into is None else 1' in source
    chain = _method_source("Qwen38TTNNLMHead", "resolve_greedy_on_device")
    assert "def resolve_greedy_on_device(self, candidates: Qwen38GreedyCandidates, *, into=None):" in chain
    assert chain.rstrip().endswith("if into is not None:\n        ttnn.copy(token_row, into)\n    return token_row")
    session = (HERE / "tools" / "qwen38_chat_session.py").read_text()
    epilogue = session[
        session.index("        def capture_epilogue(trace_output: Any)") : session.index(
            '        marker("before-chat-captures")'
        )
    ]
    assert epilogue.count("lm_head.resolve_greedy_on_device(candidates, into=token_row_io)") == 1
    # The MTP body runs the shared epilogue (its resolve into a fresh row, the MTP row, then the copy), warm and capture.
    assert epilogue.count("candidates, trace_token_row, trace_row = mtp_tail_epilogue(") == 1
    assert epilogue.index("mtp_tail_epilogue(") < epilogue.index("into=token_row_io")
    shared = session[
        session.index("def mtp_tail_epilogue(") : session.index("\n\n\n", session.index("def mtp_tail_epilogue("))
    ]
    assert (
        shared.index("token_row = lm_head.resolve_greedy_on_device(candidates)")
        < shared.index("mtp_v2.forward_mtp_step_row(")
        < shared.index("ttnn.copy(token_row, token_row_io)")
    )
    start = session.index(
        "            if chain_mtp is None:\n                candidates = lm_head.greedy_candidates(output.logits)"
    )
    warm = session[
        start : session.index("            synchronize()\n            actual = state.position.read()", start)
    ]
    assert "resolved_row = lm_head.resolve_greedy_on_device(candidates, into=token_row_io)" in warm  # the greedy body
    assert "candidates, resolved_row, warm_row = mtp_tail_epilogue(" in warm
    sampling = (HERE / "tools" / "qwen38_sampling_step.py").read_text()
    assert sampling.count("self.lm_head.resolve_greedy_on_device(candidates, into=token_row_io)") == 1


def test_lm_head_hook_and_dataclass_field_are_pinned():
    assert "    packed: Any = None  # fused greedy tail" in EMBEDDING_SOURCE
    init = _method_source("Qwen38TTNNLMHead", "__init__")
    assert 'if fused_kernels.enabled("greedy_tail"):' in init
    assert "self.greedy_candidates = functools.partial(fused_greedy_tail.greedy_candidates_fused, self)" in init
    assert (
        "self.resolve_greedy_on_device = functools.partial(fused_greedy_tail.resolve_greedy_on_device_fused, self)"
        in init
    )
    # the lanes' epilogue binds the same programs over the lane rows; the MTP rows path (resolve_greedy_rows_on_device)
    # and the chain bodies stay the chain
    assert (
        "self.greedy_candidates_lanes = functools.partial(fused_greedy_tail.greedy_candidates_lanes_fused, self)"
        in init
    )
    assert "fused_greedy_tail.resolve_greedy_lanes_on_device_fused, self" in init
    assert "resolve_greedy_rows_on_device" not in init
    for method in (
        "greedy_candidates",
        "resolve_greedy_on_device",
        "greedy_candidates_lanes",
        "resolve_greedy_lanes_on_device",
        "resolve_greedy_rows_on_device",
    ):
        assert "greedy_tail" not in _method_source("Qwen38TTNNLMHead", method)  # the chain bodies stay the chain
    lanes = _method_source("Qwen38TTNNLMHead", "greedy_candidates_lanes")
    assert lanes.rstrip().endswith("return self.greedy_candidates(logits)")
    model = (HERE / "ttnn" / "model.py").read_text()
    epilogue = model[model.index("    def resolve_lane_tokens(") : model.index("    def capture_decode_lanes(")]
    assert "candidates = lm_head.greedy_candidates_lanes(output.logits)" in epilogue
    assert "token_row = lm_head.resolve_greedy_lanes_on_device(candidates)" in epilogue


def test_lane_forms_keep_the_one_row_paths_and_the_kernel_layouts():
    """ROWS == 1 (with the two-lane packed row) is the decode step's own code path in every kernel; the lane forms
    stage one 2 KB slot per tile (scan), one 64-byte packed row per lane (merge) and one gathered row per lane
    (resolve), and the lanes' hooks gather the packed rows once."""

    scan, merge, resolve = SOURCES["scan"], SOURCES["merge"], SOURCES["resolve"]
    assert (
        "if constexpr (ROWS == 1 || LANE_SPLIT) {" in scan
        and "constexpr uint32_t LO_ROWS = ROWS < 16 ? ROWS : 16;" in scan
    )
    assert (
        "const uint32_t row = LANE_SPLIT ? get_arg_val<uint32_t>(6) : 0;" in scan
    )  # one lane row per core (lane_split)
    assert "{.page_id = r, .offset_bytes = 16 * core}" in scan  # row r of the pairs rows
    assert "if constexpr (ROWS == 1 && PACKED_LANES == 2) {" in merge  # the decode step's two-lane row keeps its path
    assert "constexpr uint32_t PACKED_ROW_BYTES = PACKED_LANES * 4;" in merge
    assert "tile[(r >> 4) * 512 + (r & 15) * 16] = best_bits >> 16;" in merge  # lane (r, 0) of the value tile
    assert "row[(r >> 4) * 256 + (r & 15)] = id;" in resolve  # lane (0, r) of the fp32 token tile
    assert "const float ranked = g[STRIDE * d] - t[d];" in resolve and "g[STRIDE * owner + 1] + s[owner]" in resolve
    assert gt.PACKED_LANES == 2 and gt.PACKED_LANES_ROWS == 16
    assert gt.merge_stage_pages(1, gt.SCAN_CORES, gt.PACKED_LANES) == gt.MERGE_STAGE_PAGES
    assert 2048 * gt.merge_stage_pages(32, 40, 16) >= 2048 + 32 * 640 + 32 * 64 + 64
    source = inspect.getsource(gt.resolve_greedy_lanes_on_device_fused)
    assert source.count("ttnn.all_gather(") == 1 and "dim=3, cluster_axis=embedding.TP_AXIS" in source
    assert "return type(lm_head).resolve_greedy_lanes_on_device(lm_head, candidates)" in source  # chain candidates
    assert "packed_lanes=PACKED_LANES_ROWS" in inspect.getsource(gt.greedy_candidates_lanes_fused)
    assert "if rows != 1:" in inspect.getsource(gt.greedy_candidates_fused)  # the MTP rows path keeps the chain
    # the decode step's packed row is the 64-byte form too: its 8-byte row made ttnn.all_gather fall back to the
    # slower composite ("input rows (8 B) are padded to the 64 B memory alignment", 2026-09-18)
    assert "greedy_candidates(logits.tensor, packed_lanes=PACKED_LANES_ROWS)" in inspect.getsource(
        gt.greedy_candidates_fused
    )


def test_manifest_lists_the_files():
    manifest_path = HERE / "tools" / "release" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("tools/release/manifest.json is not in this tree (the public tree ships without tools/release/)")
    manifest = json.loads(manifest_path.read_text())["public"]
    for path in ("tests/test_fused_greedy_tail_static.py", "ttnn/fused/greedy_tail/__init__.py") + tuple(
        f"ttnn/fused/greedy_tail/kernels/{name}.cpp" for name in ("scan", "merge", "resolve")
    ):
        assert path in manifest, path
