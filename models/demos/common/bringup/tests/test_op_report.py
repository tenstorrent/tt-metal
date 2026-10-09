# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-op report of one warm chunk (testing/op_report.py), for any spec whose hooks provide device_model.

Profiles the chunk at BRINGUP_OPREP_START (default the target's last chunk: seq - chunk) with the profiler's per-call
records, and the same layers at BRINGUP_OPREP_CONTEXT_START (default 0; "none" to skip) for the context-scaling
section. Layers (BRINGUP_OPREP_LAYERS): "rep" (default) one representative layer per spec block type, weighted by the
block type's layer count so totals are the full model's (op mode syncs after every op: the whole model is slow);
"all"; or "lo-hi". Writes <repo>/generated/<model>/op_report{.json,.txt} and prints the text.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 BRINGUP_SPEC=<spec.yaml> \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_op_report.py -s
"""

import os

import torch

from models.demos.common.bringup.core.spec import parse_layers
from models.demos.common.bringup.testing import op_report, profiler
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


def _layers_and_weights():
    sel = os.environ.get("BRINGUP_OPREP_LAYERS", "rep")
    if sel == "all":
        return S.layers(), {}
    if sel != "rep":
        lo, hi = (int(v) for v in sel.split("-"))
        return [i for i in S.layers() if lo <= i <= hi], {}
    weights = {}
    for bt, info in S.data.get("block_types", {}).items():
        members = [i for i in parse_layers(info["layers"], S.num_layers) if i in S.layers()]
        if members:
            weights[S.representative_layer(bt)] = len(members)
    return sorted(weights), weights


def _axis_links(mesh_device) -> dict:
    """Fabric links per mesh axis: BRINGUP_OPREP_AXIS_LINKS="a0,a1", else the system table the DeepSeek-family CCLs
    use (P150x8 LoudBox: 2 and 2), else 1."""
    env = os.environ.get("BRINGUP_OPREP_AXIS_LINKS")
    if env:
        return {i: int(v) for i, v in enumerate(env.split(","))}
    try:
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links

        return {a: int(get_num_links(mesh_device, a)) for a in range(len(tuple(mesh_device.shape)))}
    except Exception:
        return {}


@mesh_parametrize
def test_op_report(mesh_device):
    import ttnn
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens

    for k, v in profiler.PROFILER_ENV.items():
        assert os.environ.get(k) == v, f"set {k}={v}"
    seq, chunk = int(S.get("target.seq")), int(S.get("target.chunk"))
    start = int(os.environ.get("BRINGUP_OPREP_START", seq - chunk))
    ctx_env = os.environ.get("BRINGUP_OPREP_CONTEXT_START", "0")
    ctx_start = None if ctx_env == "none" else int(ctx_env)
    layers, weights = _layers_and_weights()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    all_toks = prompt_tokens(S, seq).to(torch.long)

    def run(s0):
        h = model.embed(all_toks[s0 : s0 + chunk])
        for i in layers:
            profiler.set_layer(i)
            h2 = model.layer(i, h, s0, None)
            model.free(h)
            h = h2
        profiler.set_layer(None)
        model.free(h)

    def profile_at(s0):
        run(s0)  # compile / warm this position's programs
        model.sync()
        profiler.enable(mesh_device, ops=True, calls=True)
        try:
            run(s0)
            profiler.signpost("end")
            return profiler.result()["calls"]
        finally:
            profiler.disable()

    calls = profile_at(start)
    ctx_calls = profile_at(ctx_start) if ctx_start is not None and ctx_start != start else None
    grid = mesh_device.compute_with_storage_grid_size()
    axis_links = _axis_links(mesh_device)
    rep = op_report.build(
        calls,
        tuple(mesh_device.shape),
        (grid.x, grid.y),
        arch=ttnn.get_arch_name(),
        layer_weights=weights,
        context_calls=ctx_calls,
        context_label=f"start {ctx_start}" if ctx_calls is not None else "",
        axis_links=axis_links,
    )
    title = f"{S.model}: chunk [{start}, {start + chunk}), {len(layers)} profiled layers"
    text = op_report.render(rep, title=title)
    out = S.repo / "generated" / S.model
    out.mkdir(parents=True, exist_ok=True)
    op_report.save(rep, out / "op_report.json")
    op_report.save_calls(  # the raw per-call records: re-render offline with python -m ...testing.op_report
        out / "op_calls.json",
        calls=calls,
        context_calls=ctx_calls,
        mesh=tuple(mesh_device.shape),
        grid=(grid.x, grid.y),
        arch=ttnn.get_arch_name(),
        layer_weights=weights,
        axis_links=axis_links,
        context_label=f"start {ctx_start}" if ctx_calls is not None else "",
        title=title,
    )
    (out / "op_report.txt").write_text(text + "\n")
    print(text, flush=True)
    print(f"wrote {out / 'op_report.json'} and op_report.txt", flush=True)
    assert rep["rows"], "no profiled ops"
