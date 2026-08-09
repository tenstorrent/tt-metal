# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-pipeline dry run: connect the *real* MiniMax-H3 text encoder -> DiT.

Both stages are built on one mesh (as the real pipeline does), dry-run into their own
graphs, and linked with connect=True so the encoder's text embeds feed the DiT's prompt
across the readback boundary. The encoder output is unsqueezed to [1,1,L,dim] to match the
DiT's prompt_1BLP, mirroring the pipeline's own handoff.

Usage:  python3 scout_h3_pipeline.py [2x4|4x8]   (default 4x8)

The headline finding is `replicated_stage`: the TP-only encoder replicates across the SP
axis, so its output is identical on every SP row but the handoff consumes one -- 1 of 2
rows wasted at 2x4, 7 of 8 at 4x8. It is clean analysed standalone; the finding appears
only once the stages are connected.
"""
import os
import sys

TOOLS = "models/tt_dit/tools"
sys.path.insert(0, TOOLS)

from dit_analyzer import analyze_graph  # noqa: E402
from dit_analyzer.dryrun import install, recorder, start  # noqa: E402
from dit_analyzer.dryrun.context import CTX  # noqa: E402
from dit_analyzer.dryrun.weights import load_meta_weights  # noqa: E402
from dit_analyzer.ir import Dist  # noqa: E402
from dit_analyzer.link import link_stages  # noqa: E402
from dit_analyzer.report import render_report  # noqa: E402

# preset -> (mesh, sp_axis, tp_axis, text_len, num_audio, num_video)
PRESETS = {
    "2x4": ((2, 4), 0, 1, 128, 64, 192),
    "4x8": ((4, 8), 1, 0, 512, 256, 1280),
    "prod": ((4, 8), 1, 0, 512, 414, 37296),  # production 768p / 5s working point (real modality counts)
}
preset = sys.argv[1] if len(sys.argv) > 1 else "4x8"
MESH, SP_AXIS, TP_AXIS, L, NUM_AUDIO, NUM_VIDEO = PRESETS[preset]

# Depth. Every byte count this tool reports is per *forward*, so a stack built one layer deep
# reports one layer's traffic — a floor, not the number. Production MiniMax-H3 is 50 DiT layers
# (tests/models/minimax_h3/project_block_perf.py: DEFAULT_LAYERS) and a text conditioner truncated
# at the tapped layer, MINIMAX_H3_TEXT_ENCODER_LAYER = 50 (pipelines/minimax_h3/packing.py), which
# is exactly what pipeline_minimax_h3 builds. The refiner's 2 is already the real value.
# `prod` therefore runs real depth; the smoke presets stay at 1 so they stay seconds-long.
# Override per stage with DITCHECK_ENC_LAYERS / DITCHECK_DIT_LAYERS / DITCHECK_REFINER_LAYERS —
# analysis cost grows with depth, so bisecting it is often what you want.
# How often each stage runs in ONE generation. The prompt is encoded once and the VAEs decode
# once; only the DiT sits inside the denoise loop. pipeline_minimax_h3 defaults to
# num_inference_steps=50, and scheduler.py is explicit that the count is grid points and drives
# `num_inference_steps - 1` model evaluations -- so 49 DiT forwards, not 50. Without this the
# report adds a once-per-generation encoder to a per-step DiT as if they ran equally often.
NUM_INFERENCE_STEPS = int(os.environ.get("DITCHECK_STEPS", 50))
DIT_EVALS = max(1, NUM_INFERENCE_STEPS - 1)
STAGE_STEPS = {"encoder": 1, "dit": DIT_EVALS, "vae": 1, "audio_vae": 1}

_DEPTH = {"prod": (50, 50, 2)}.get(preset, (1, 1, 1))
ENC_LAYERS = int(os.environ.get("DITCHECK_ENC_LAYERS", _DEPTH[0]))
DIT_LAYERS = int(os.environ.get("DITCHECK_DIT_LAYERS", _DEPTH[1]))
REFINER_LAYERS = int(os.environ.get("DITCHECK_REFINER_LAYERS", _DEPTH[2]))
HID = 5120  # encoder hidden == DiT text_dim (the real contract)
SEQ = L + NUM_AUDIO + NUM_VIDEO
ALIGN = MESH[SP_AXIS] * 32  # the DiT pads the packed sequence to a multiple of sp_factor * TILE
PADDED = -(-SEQ // ALIGN) * ALIGN
print(
    "preset %s: mesh %s  sp=axis%d tp=axis%d  text/audio/video=%d/%d/%d  packed_seq=%d (padded %d)"
    % (preset, MESH, SP_AXIS, TP_AXIS, L, NUM_AUDIO, NUM_VIDEO, SEQ, PADDED)
)
print(
    "depth: encoder %d, dit %d, refiner %d   |   %d inference steps -> %d DiT evaluations "
    "(encoder and VAEs run once per generation)"
    % (ENC_LAYERS, DIT_LAYERS, REFINER_LAYERS, NUM_INFERENCE_STEPS, DIT_EVALS)
)
# Axis names must follow the preset, not the 2x4 habit: at 4x8 (and prod) sp is axis1, so a
# hardcoded ("sp", "tp") labels every finding's mesh_axis with the *other* parallelism. That
# doesn't change any verdict or byte count (those come from the participant groups) but it
# points a reader at the wrong axis, which is worse than saying nothing.
AXIS_NAMES = ("sp", "tp") if SP_AXIS == 0 else ("tp", "sp")

mesh_device = install(MESH, "blackhole")
import ttnn  # noqa: E402
from models.tt_dit.encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder  # noqa: E402
from models.tt_dit.models.transformers.minimax_h3.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel,
)
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor  # noqa: E402
from models.tt_dit.parallel.manager import CCLManager  # noqa: E402

rep = lambda shp: recorder.entry(shp, Dist.replicated(CTX.mesh), base="in")  # noqa: E731
repf = lambda shp: recorder.entry(shp, Dist.replicated(CTX.mesh), dtype=ttnn.float32, base="in")  # noqa: E731

# ---- stage 1: text encoder -> [1, 1, L, HID] (unsqueezed to match the DiT prompt) ----
enc_graph = start(mesh_device, axis_names=AXIS_NAMES, name="encoder", topology="Linear")
enc_ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
enc_pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=MESH[TP_AXIS], mesh_axis=TP_AXIS))
enc = Qwen3VlTextEncoder(
    vocab_size=151936,
    hidden_size=HID,
    intermediate_size=25600,
    hidden_act="silu",
    num_hidden_layers=ENC_LAYERS,
    num_attention_heads=64,
    num_key_value_heads=8,
    rms_norm_eps=1e-6,
    rope_theta=1e7,
    mrope_section=[16, 24, 24],
    head_dim=128,
    device=mesh_device,
    parallel_config=enc_pc,
    ccl_manager=enc_ccl,
)
load_meta_weights(enc)
taps = enc.forward(rep([1, L]), pos_embeds=(repf([1, 1, L, 128]), repf([1, 1, L, 128])))
embeds = ttnn.unsqueeze(taps[0] if isinstance(taps, list) else taps, 1)  # [1, L, HID] -> [1, 1, L, HID]
enc_graph.outputs = [embeds.sym]
print("encoder stage: %d nodes, output %s" % (len(enc_graph.nodes), list(enc_graph.symbol(embeds.sym).shape)))

# ---- stage 2: DiT (its prompt_1BLP is [1, 1, L, text_dim=HID] -> the handoff) ----
dit_graph = start(mesh_device, axis_names=AXIS_NAMES, name="dit", topology="Ring")
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
    num_layers=DIT_LAYERS,
    num_refiner_layers=REFINER_LAYERS,
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
    prompt_1BLP=rep([1, 1, L, HID]),  # <- fed by the encoder once linked
    timestep=repf([1, 1, 2, 1]),
    adaln_indices=recorder.entry([1, 1, 1, PADDED], sp_last, base="adaln"),
    timestep_indices=recorder.entry([1, 1, 1, PADDED], sp_last, base="tsi"),
    rope_cos=recorder.entry([1, 1, PADDED, 128], sp_seq, dtype=ttnn.float32, base="rcos"),
    rope_sin=recorder.entry([1, 1, PADDED, 128], sp_seq, dtype=ttnn.float32, base="rsin"),
)
dit_graph.outputs = [v_out.sym, a_out.sym]
print("dit stage: %d nodes" % len(dit_graph.nodes))

# ---- stage 3: VAE ViT decoder (video), fed by the *real* unpatchify of the DiT video output ----
# The pipeline reads the DiT video velocity back to host and unpatchifies it (packing.py
# unpatchify_video_tokens): [1,1,V,96] --reshape/permute--> latents [1,24,T,H,W], which the VAE
# decodes as num_patches = T*H*W voxels. The DiT patch (1,2,2) expands V patched rows to V*4
# voxels (96 -> 24 channels). At prod this is the real 768p/5s latent geometry pulled from the
# full pipeline test (models/tt_dit/tests/.../test_pipeline_minimax_h3.py: 124f, 768x1344):
# video_latent_num_frames(124)=37, 768//16=48, 1344//16=84 -> [1,24,37,48,84] -> [1,149184,24].
# The ViT decoder holds NO CCL (one tile per device, SPMD), so a single untiled unit gives the
# same collective picture as the real tiled decode; only the boundary geometry needs to be real.
from models.tt_dit.models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3ViTDecoder3d  # noqa: E402

VAE_LAT = (37, 48, 84) if preset == "prod" else (7, 16, 16)  # latent (T, H, W); prod = real unpatchify geometry
vf, vh, vw = VAE_LAT
vae_graph = start(mesh_device, axis_names=AXIS_NAMES, name="vae", topology="Linear")
vae = MiniMaxH3ViTDecoder3d(
    num_frames=vf,
    height=vh,
    width=vw,
    in_channels=24,
    out_channels=3,
    num_layers=1,
    num_heads=32,
    head_dim=64,
    num_register_tokens=4,
    rope_theta=100.0,
    rope_dim_ratio=0.75,
    eps=1e-5,
    mesh_device=mesh_device,
)
load_meta_weights(vae)
tokens = rep([1, vf * vh * vw, 24])  # unpatchified latent voxels <- DiT video output across the boundary
vae_graph.outputs = [vae(tokens).sym]
print("vae stage: %d nodes  (latent %dx%dx%d -> %d voxel tokens)" % (len(vae_graph.nodes), vf, vh, vw, vf * vh * vw))

# ---- stage 4: audio VAE decoder (HiFi-GAN-style vocoder), fed by the DiT *audio* output ----
# The pipeline reads the DiT audio velocity back to host and unpacks it (packing.unpack_audio_tokens):
# [NUM_AUDIO, 32] --reshape/permute--> [2, 32, T] (stereo B=2, latent_channels C=32, T audio latents),
# then audio_decoder(latents_BCT). NUM_AUDIO = num_audio_latents * 2, so T = NUM_AUDIO // 2 (prod: 207).
# The audio decoder's __init__ defaults *are* the real config; no snapshot needed. Like the video
# decoder it carries no parallel_config, so it too is collective-free -- wiring it exists to make the
# DiT audio output a *consumed* handoff (else it reads back as a dead_collective, an unwired-branch artifact).
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder  # noqa: E402

AUDIO_T = NUM_AUDIO // 2
avae_graph = start(mesh_device, axis_names=AXIS_NAMES, name="audio_vae", topology="Linear")
adec = MiniMaxH3AudioDecoder(mesh_device=mesh_device)  # real config defaults (latent 32, dim 2048, ...)
load_meta_weights(adec)
# The audio decoder is a host-orchestrated HiFi-GAN vocoder: each stage (dec_in_proj, resample,
# vocoder) owns its own upload/readback, and it carries no parallel_config, so it is collective-free
# end to end. The first device op the unpacked latents hit is dec_in_proj (a k1 conv 32->2048); wiring
# the handoff to it makes the DiT audio output a *consumed* boundary (resolving the node_360
# dead_collective) without dry-running the whole collective-free vocoder. Device input is [B, T, C]
# (unpack_audio_tokens then the decoder's own transpose(1,2)): [2, T=NUM_AUDIO//2, 32].
audio_in = recorder.entry(  # [B=2 stereo, T, C=32], row-major (the audio conv1d requires it)
    [2, AUDIO_T, 32], Dist.replicated(CTX.mesh), base="in", layout=ttnn.ROW_MAJOR_LAYOUT
)
avae_graph.outputs = [adec.dec_in_proj(audio_in).sym]
print("audio_vae stage: %d nodes  (audio latents [2, %d, 32] -> dec_in_proj)" % (len(avae_graph.nodes), AUDIO_T))

# ---- link the four real stages as a DAG: the DiT fans out to the video VAE (output 0) and the
#      audio VAE (output 1); both bridge a host reshape/unpack across the readback boundary ----
linked = link_stages(
    [
        ("encoder", enc_graph),
        ("dit", dit_graph),
        ("vae", vae_graph, tokens.sym),  # source defaults to the DiT's video output (index 0)
        ("audio_vae", avae_graph, audio_in.sym, ("dit", 1)),  # <- the DiT's audio output (index 1)
    ],
    connect=True,
    stage_steps=STAGE_STEPS,
)
print("\n" + render_report(analyze_graph(linked), top=8, proof=False))
