"""Trace overwide_gather / participant_shrink in the H3 DiT stage, standalone.

Builds the real encoder+DiT (prod dims) exactly as the scout does, but analyses the
DiT graph on its own — no VAE, no linking. A finding that appears here is intrinsic to
the DiT's collective structure, not an artifact of the stage boundary.
"""
import json
import sys

TOOLS = "models/tt_dit/tools"
sys.path.insert(0, TOOLS)

from dit_analyzer import analyze_graph  # noqa: E402
from dit_analyzer.dryrun import install, recorder, start  # noqa: E402
from dit_analyzer.dryrun.context import CTX  # noqa: E402
from dit_analyzer.dryrun.weights import load_meta_weights  # noqa: E402
from dit_analyzer.ir import Dist  # noqa: E402
from dit_analyzer.report import render_report  # noqa: E402

MESH, SP_AXIS, TP_AXIS, L, NUM_AUDIO, NUM_VIDEO = (4, 8), 1, 0, 512, 414, 37296
HID = 5120
SEQ = L + NUM_AUDIO + NUM_VIDEO
ALIGN = MESH[SP_AXIS] * 32
PADDED = -(-SEQ // ALIGN) * ALIGN

mesh_device = install(MESH, "blackhole")
import ttnn  # noqa: E402
from models.tt_dit.models.transformers.minimax_h3.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel,
)
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor  # noqa: E402
from models.tt_dit.parallel.manager import CCLManager  # noqa: E402

rep = lambda shp: recorder.entry(shp, Dist.replicated(CTX.mesh), base="in")  # noqa: E731
repf = lambda shp: recorder.entry(shp, Dist.replicated(CTX.mesh), dtype=ttnn.float32, base="in")  # noqa: E731

dit_graph = start(mesh_device, axis_names=("sp", "tp"), name="dit", topology="Ring")
dit_ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
dit_pc = DiTParallelConfig(
    tensor_parallel=ParallelFactor(mesh_axis=TP_AXIS, factor=MESH[TP_AXIS]),
    sequence_parallel=ParallelFactor(mesh_axis=SP_AXIS, factor=MESH[SP_AXIS]),
    cfg_parallel=None,
)
dit = MiniMaxH3Transformer3DModel(
    num_attention_heads=56,
    attention_head_dim=128,
    hidden_size=5376,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=14336,
    in_channels=24,
    audio_in_channels=32,
    patch_size=(1, 2, 2),
    text_dim=HID,
    freq_dim=256,
    time_embed_hidden_dim=5376,
    time_embed_dim=2688,
    rope_freq_dim=16,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
    mesh_device=mesh_device,
    ccl_manager=dit_ccl,
    parallel_config=dit_pc,
    is_fsdp=False,
)
load_meta_weights(dit)
sp_seq = Dist.make(CTX.mesh, {SP_AXIS: 2})
sp_last = Dist.make(CTX.mesh, {SP_AXIS: 3})
v_out, a_out = dit.forward(
    video_1BVC=rep([1, 1, NUM_VIDEO, 96]),
    audio_1BAC=rep([1, 1, NUM_AUDIO, 32]),
    prompt_1BLP=rep([1, 1, L, HID]),
    timestep=repf([1, 1, 2, 1]),
    adaln_indices=recorder.entry([1, 1, 1, PADDED], sp_last, base="adaln"),
    timestep_indices=recorder.entry([1, 1, 1, PADDED], sp_last, base="tsi"),
    rope_cos=recorder.entry([1, 1, PADDED, 128], sp_seq, dtype=ttnn.float32, base="rcos"),
    rope_sin=recorder.entry([1, 1, PADDED, 128], sp_seq, dtype=ttnn.float32, base="rsin"),
)
dit_graph.outputs = [v_out.sym, a_out.sym]
print("dit stage: %d nodes" % len(dit_graph.nodes))

rep_out = analyze_graph(dit_graph)
print("\n" + render_report(rep_out, top=12, proof=False))

by_id = {n.id: n for n in dit_graph.nodes}


def consumers(o):
    return [(n.id, n.op) for n in dit_graph.nodes if o in n.inputs]


def sh(sid):
    s = dit_graph.symbols.get(sid)
    return list(s.shape) if s else "?"


hits = [f for f in rep_out.findings if f.rule in ("overwide_gather", "participant_shrink")]
print("\n########## TRACE: %d overwide/participant (DiT standalone) ##########" % len(hits))
for i, f in enumerate(hits):
    print("\n===== [%d] %s (%s/%s) =====" % (i, f.rule, f.severity, f.confidence))
    print("title:", f.title)
    for nid in f.nodes:
        n = by_id.get(nid)
        if not n:
            continue
        print("  NODE %s op=%s mesh_axis=%s loc=%s" % (n.id, n.op, n.mesh_axis, n.loc))
        print("     label=%s" % n.label)
        print("     inputs=%s" % [(x, sh(x)) for x in n.inputs])
        print("     outputs=%s" % [(x, sh(x)) for x in n.outputs])
        for o in n.outputs:
            print("     consumers: %s" % consumers(o))
    print("  reason:")
    for r in f.reason:
        print("   -", r)
    print("  suggestion:", f.suggestion)
    print("  proof:", json.dumps(f.proof, indent=2, default=str))
