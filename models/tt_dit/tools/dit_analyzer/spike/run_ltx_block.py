# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Spike driver: run the real LTX-2.3 block under the fake ttnn, analyze the result.

    python3 models/tt_dit/tools/dit_analyzer/spike/run_ltx_block.py [--ring|--linear]

Success criteria (roadmap phase 6 acceptance, in miniature):
  1. the real forward completes under the shim
  2. per-device shapes match the hand-written oracle, so shape-dependent branches
     (attention_ltx.py:483) take the same path
  3. the collectives match examples/ltx.py: count, axis, dim, shapes
  4. the analyzer reports the same 6 duplicate_gather findings
"""

from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(TOOLS)))
sys.path.insert(0, TOOLS)
sys.path.insert(0, REPO)

import fake_torch  # noqa: E402
import fake_ttnn  # noqa: E402


def install_py310_annotation_hook() -> None:
    """Compile tt_dit sources with `from __future__ import annotations`.

    The repo targets Python >= 3.10 and uses PEP 604 unions (`ttnn.Tensor | None`)
    in *evaluated* annotation positions; this box has 3.9. Stringifying annotations
    at compile time sidesteps that. A real dry run just uses the repo's own
    interpreter and does not need this.
    """
    import __future__
    import importlib.abc
    import importlib.machinery

    flag = __future__.annotations.compiler_flag

    sys.dont_write_bytecode = True  # and never read a .pyc compiled without the flag

    class _Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            source = self.get_data(self.get_filename(fullname))
            return compile(source, self.get_filename(fullname), "exec", dont_inherit=True, flags=flag)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("models.tt_dit"):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                return spec
            spec.loader = _Loader(spec.loader.name, spec.loader.path)
            return spec

    sys.meta_path.insert(0, _Finder())


def install_host_fakes() -> None:
    """torch / loguru / safetensors / models.common: only what the import path needs."""
    install_py310_annotation_hook()
    if not hasattr(types, "NoneType"):  # 3.10+; utils/tracing.py imports it
        types.NoneType = type(None)
    fake_torch.install()

    loguru = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru

    st = types.ModuleType("safetensors")
    st.safe_open = lambda *a, **k: None
    st_torch = types.ModuleType("safetensors.torch")
    st_torch.load_file = lambda *a, **k: {}
    st.torch = st_torch
    sys.modules["safetensors"] = st
    sys.modules["safetensors.torch"] = st_torch

    # models.common.utility_functions drags in numpy + pytest; tt_dit only wants
    # is_blackhole from it.
    for name in ("models", "models.common"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(REPO, *name.split("."))]
            sys.modules[name] = mod
    uf = types.ModuleType("models.common.utility_functions")
    uf.is_blackhole = lambda *a, **k: True
    uf.is_wormhole_b0 = lambda *a, **k: False
    uf.nearest_32 = lambda x: -(-x // 32) * 32
    sys.modules["models.common.utility_functions"] = uf


# LTX-2.3 shapes, same as examples/ltx.py
VIDEO_DIM, VIDEO_HEADS, VIDEO_HEAD_DIM, VIDEO_N = 4096, 32, 128, 38912
AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM, AUDIO_N = 2048, 32, 64, 256
TEXT_L, LAYERS, STEPS = 32, 48, 8
VIDEO_FFN, AUDIO_FFN = 4 * VIDEO_DIM, 4 * AUDIO_DIM


def build_and_run(ring: bool = True):
    install_host_fakes()
    mesh_shape = (4, 8) if ring else (2, 4)
    mesh_device = fake_ttnn.install(mesh_shape=mesh_shape, arch="blackhole")
    import ttnn  # the fake

    tp_axis, sp_axis = 0, 1  # pipeline_ltx.py: sp_axis=1, tp_axis=0 on both BH configs
    graph = fake_ttnn.REC.start(
        mesh_device,
        axis_names=("tp", "sp"),
        name="ltx_dryrun_bh_%dx%d" % mesh_shape,
        steps=STEPS,
        model="LTX-2.3 (audio + video), dry run",
        note="one LTXTransformerBlock, x%d layers; %s topology" % (LAYERS, "Ring" if ring else "Linear"),
    )
    fake_ttnn.REC.calls = LAYERS

    from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerBlock  # noqa: E402
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor  # noqa: E402
    from models.tt_dit.parallel.manager import CCLManager  # noqa: E402

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
    )
    ccl = CCLManager(
        mesh_device,
        num_links=2,
        topology=ttnn.Topology.Ring if ring else ttnn.Topology.Linear,
    )

    block = LTXTransformerBlock(
        video_dim=VIDEO_DIM,
        audio_dim=AUDIO_DIM,
        video_ffn_dim=VIDEO_FFN,
        audio_ffn_dim=AUDIO_FFN,
        video_num_heads=VIDEO_HEADS,
        audio_num_heads=AUDIO_HEADS,
        video_cross_attention_dim=VIDEO_DIM,
        audio_cross_attention_dim=AUDIO_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        parallel_config=parallel_config,
        has_audio=True,
        apply_gated_attention=True,
        cross_attention_adaln=True,
    )
    inject_weights(block, tp_axis)
    out = run_forward(block, tp_axis, sp_axis)
    graph.outputs = [t.sym for t in out if t is not None]
    return graph


def inject_weights(module, tp_axis: int, prefix: str = "") -> int:
    """Give every Parameter metadata-only data, as meta-tensor weights would.

    `attention_ltx.py:379` gates the gate projection on `weight._data`, so a run
    with unloaded weights would silently drop the finding we are looking for.
    """
    from dit_analyzer.ir import PARAM, Dist

    import fake_ttnn as ft

    count = 0
    for name, p in module.named_parameters():
        shard = {}
        for tensor_dim, mesh_axis in enumerate(p.mesh_axes or []):
            if mesh_axis is not None:
                shard[mesh_axis] = tensor_dim
        dist = Dist.make(ft.REC.mesh, shard)
        p._data = ft.REC.entry(list(p.total_shape), dist, p.dtype, kind=PARAM, base=(prefix + name).replace(".", "_"))
        count += 1
    for name, child in module.named_children():
        count += inject_weights(child, tp_axis, prefix + name + ".")
    return count


def run_forward(block, tp_axis: int, sp_axis: int):
    import fake_ttnn as ft
    from dit_analyzer.ir import Dist

    def tensor(logical, shard=None):
        return ft.REC.entry(logical, Dist.make(ft.REC.mesh, shard or {}), base="input")

    sp_n, tp_d = {sp_axis: 2}, {tp_axis: 3}
    act = dict(sp_n)
    act.update(tp_d)

    video = tensor([1, 1, VIDEO_N, VIDEO_DIM], act)
    audio = tensor([1, 1, AUDIO_N, AUDIO_DIM], act)
    video_prompt = tensor([1, 1, TEXT_L, VIDEO_DIM])  # replicated text embeddings
    audio_prompt = tensor([1, 1, TEXT_L, AUDIO_DIM])
    adaln = 9  # cross_attention_adaln=True
    video_temb = tensor([adaln, 1, 1, VIDEO_DIM], tp_d)
    audio_temb = tensor([adaln, 1, 1, AUDIO_DIM], tp_d)
    video_prompt_temb = tensor([2, 1, 1, VIDEO_DIM])
    audio_prompt_temb = tensor([2, 1, 1, AUDIO_DIM])
    av_ca_temb = tensor([5, 1, 1, VIDEO_DIM], tp_d)
    av_ca_audio_temb = tensor([5, 1, 1, AUDIO_DIM], tp_d)
    trans_mat = tensor([1, 1, 32, 32])

    return block(
        video_1BND=video,
        video_prompt=video_prompt,
        video_temb=video_temb,
        video_N=VIDEO_N,
        video_rope_cos=tensor([1, 1, VIDEO_N, VIDEO_HEAD_DIM], {sp_axis: 2}),
        video_rope_sin=tensor([1, 1, VIDEO_N, VIDEO_HEAD_DIM], {sp_axis: 2}),
        trans_mat=trans_mat,
        video_prompt_temb=video_prompt_temb,
        audio_1BND=audio,
        audio_prompt=audio_prompt,
        audio_temb=audio_temb,
        av_ca_temb=av_ca_temb,
        audio_N=AUDIO_N,
        audio_rope_cos=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], {sp_axis: 2}),
        audio_rope_sin=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], {sp_axis: 2}),
        av_ca_audio_temb=av_ca_audio_temb,
        audio_prompt_temb=audio_prompt_temb,
        video_cross_pe_cos=tensor([1, 1, VIDEO_N, AUDIO_HEAD_DIM], {sp_axis: 2}),
        video_cross_pe_sin=tensor([1, 1, VIDEO_N, AUDIO_HEAD_DIM], {sp_axis: 2}),
        audio_cross_pe_cos=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], {sp_axis: 2}),
        audio_cross_pe_sin=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], {sp_axis: 2}),
        audio_cross_pe_cos_full=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM]),
        audio_cross_pe_sin_full=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM]),
        audio_attn_mask=tensor([1, 1, AUDIO_N, AUDIO_N]),
        audio_padding_mask_full=tensor([1, 1, AUDIO_N, 1]),
    )


def compare_with_oracle(graph, ring: bool):
    """Criterion 3/4: collectives and findings vs the hand-written example."""
    from dit_analyzer import analyze_graph
    from dit_analyzer.examples import load
    from dit_analyzer.semantics import lookup

    oracle = load("ltx_block_bh_4x8" if ring else "ltx_block_bh_2x4")

    def collectives(g):
        """(op, mesh axis, gathered axis size, non-unit logical extents).

        Rank-normalised: the hand model uses [1, N, D] where the real code uses
        [1, B, N, D], so compare the extents that carry information.
        """
        out = []
        for n in g.nodes:
            if lookup(n.op).is_collective:
                shape = tuple(d for d in g.symbol(n.outputs[0]).shape if d != 1)
                dim = n.attrs.get("dim")
                extent = g.symbol(n.outputs[0]).shape[dim] if dim is not None else None
                out.append((n.op, n.mesh_axis, extent, shape))
        return out

    dry, ref = collectives(graph), collectives(oracle)
    only_dry: list = []
    only_ref: list = []
    print("\ncollectives: dry run %d, oracle %d" % (len(dry), len(ref)))
    from collections import Counter

    cd, cr = Counter(dry), Counter(ref)
    only_dry[:] = sorted((cd - cr).elements())
    only_ref[:] = sorted((cr - cd).elements())
    if only_dry or only_ref:
        print("  only in dry run:", len(only_dry))
        for x in only_dry[:12]:
            print("    +", x)
        print("  only in oracle:", len(only_ref))
        for x in only_ref[:12]:
            print("    -", x)
    else:
        print("  identical multiset of (op, axis, dim, shape)")

    report = analyze_graph(graph)
    rules = Counter(f.rule for f in report.findings)
    print("\nfindings: %s" % (dict(rules) or "none"))
    for f in report.findings[:8]:
        print("  [%s/%s] %s" % (f.severity, f.confidence, f.title))
        if f.loc:
            print("        %s" % f.loc)
    if report.diagnostics:
        codes = Counter(d.code for d in report.diagnostics)
        print("diagnostics:", dict(codes))

    oracle_report = analyze_graph(oracle)
    failures = check_criteria(graph, report, oracle_report, only_dry, only_ref)
    return report, failures


def check_criteria(graph, report, oracle_report, only_dry, only_ref):
    """The four acceptance criteria, as assertions. Empty list == pass."""
    fail = []
    if not graph.nodes:
        fail.append("1: dry run produced no nodes")
    if fake_ttnn.UNREGISTERED:
        fail.append("1: unregistered ops: %s" % sorted(fake_ttnn.UNREGISTERED))
    if report.diagnostics:
        fail.append("2: analyzer diagnostics: %s" % sorted({d.code for d in report.diagnostics}))
    if only_dry or only_ref:
        fail.append("3: collectives differ from the oracle (+%d / -%d)" % (len(only_dry), len(only_ref)))

    def signature(r):
        return sorted((f.rule, f.confidence, f.bytes_per_call, f.calls) for f in r.findings)

    if signature(report) != signature(oracle_report):
        fail.append(
            "4: findings differ from the oracle\n     dry run: %s\n     oracle:  %s"
            % (signature(report), signature(oracle_report))
        )
    return fail


def main() -> int:
    ring = "--linear" not in sys.argv
    graph = build_and_run(ring=ring)
    print("=" * 78)
    print("dry run OK: %d nodes, %d symbols, mesh %s" % (len(graph.nodes), len(graph.symbols), graph.mesh.shape))
    if fake_ttnn.UNREGISTERED:
        print("\nunregistered ops encountered (would be `ops --missing` output):")
        for name, count in sorted(fake_ttnn.UNREGISTERED.items(), key=lambda kv: -kv[1]):
            print("  %-60s x%d" % (name, count))
    else:
        print("unregistered ops: none")
    _, failures = compare_with_oracle(graph, ring)
    out = os.path.join(HERE, "ltx_dryrun.graph.json")
    with open(out, "w") as fh:
        fh.write(graph.to_json())
    print("\nwrote %s" % out)
    if failures:
        print("\nFAIL (%d criteria):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("\nPASS: dry run matches the hand-written oracle on all four criteria")
    return 0


if __name__ == "__main__":
    sys.exit(main())
