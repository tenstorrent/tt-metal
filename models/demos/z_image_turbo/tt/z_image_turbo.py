# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo pipeline with Metal Trace on TE, DIT, and VAE.

Metal Trace eliminates host dispatch overhead by recording the TTNN op graph once
and replaying it from device-side command buffers.  All three models are traced:
text encoder (1 call), DIT (8 calls), and VAE decoder (1 call).
"""

import copy
import gc
import time

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.z_image_turbo.tt.dit.model_ttnn import ZImageTransformerTTNN
from models.demos.z_image_turbo.tt.text_encoder.model_ttnn import TextEncoderTTNN
from models.demos.z_image_turbo.tt.vae.model_ttnn import VaeDecoderTTNN

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _dram_stats(mesh_device, label):
    """Print DRAM allocation stats."""
    ttnn.synchronize_device(mesh_device)
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    alloc = v.num_banks * v.total_bytes_allocated_per_bank
    total = v.num_banks * v.total_bytes_per_bank
    free = v.num_banks * v.total_bytes_free_per_bank
    print(f"  DRAM [{label}]: {alloc / 2**20:.1f} MB used / {total / 2**20:.1f} MB total ({free / 2**20:.1f} MB free)")


# ── Tensor helpers ────────────────────────────────────────────────────────────


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


def _copy_to_persistent(host_pt, persistent_tt, dtype=ttnn.DataType.BFLOAT16):
    """Write a CPU tensor into every per-device shard of a persistent mesh tensor.

    Uses per-device iteration so that all 4 BH devices get the new data
    (plain copy_host_to_device_tensor only reaches device 0).
    """
    host_tt = ttnn.from_torch(
        host_pt.bfloat16() if dtype == ttnn.DataType.BFLOAT16 else host_pt,
        dtype=dtype,
        layout=ttnn.Layout.ROW_MAJOR,
    )
    for shard in ttnn.get_device_tensors(persistent_tt):
        ttnn.copy_host_to_device_tensor(host_tt, shard, cq_id=0)


def _copy_to_persistent_device(src_tt, persistent_tt):
    """Copy a device mesh tensor into a persistent device mesh tensor (same addresses).

    Both tensors must have the same shape, dtype, and layout.
    """
    src_host = ttnn.from_device(src_tt)
    for src_shard, dst_shard in zip(ttnn.get_device_tensors(src_host), ttnn.get_device_tensors(persistent_tt)):
        ttnn.copy_host_to_device_tensor(src_shard, dst_shard, cq_id=0)


# ── Scheduler helper ─────────────────────────────────────────────────────────


def _compute_mu(
    h=IMG_LATENT_H,
    w=IMG_LATENT_W,
    base_seq=256,
    max_seq=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    seq = (h // 2) * (w // 2)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return seq * m + (base_shift - m * base_seq)


def _make_scheduler(template, steps):
    scheduler = copy.deepcopy(template)
    mu = _compute_mu()
    try:
        scheduler.set_timesteps(steps, mu=mu)
    except TypeError:
        scheduler.set_timesteps(steps)
    return scheduler


# ── ZImageTurbo class ─────────────────────────────────────────────────────────


class ZImageTurbo(LightweightModule):
    """Z-Image-Turbo pipeline with Metal Trace on TE, DIT, and VAE."""

    DEFAULT_MESH_SHAPE = (1, 4)

    def __init__(self, mesh_device=None):
        t0 = time.time()

        print("[1/5] Loading CPU components ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        self.vae_processor = VaeImageProcessor(vae_scale_factor=16)
        self.scheduler_template = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        self.scheduler_template.sigma_min = 0.0

        if mesh_device is None:
            print(f"[2/5] Opening TTNN {self.DEFAULT_MESH_SHAPE} mesh device ...")
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            self.mesh_device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*self.DEFAULT_MESH_SHAPE),
            )
        else:
            print("[2/5] Using caller-provided mesh device ...")
            self.mesh_device = mesh_device
        self.mesh_device.enable_program_cache()

        print("[3/5] Building Text Encoder ...")
        self.te = TextEncoderTTNN(self.mesh_device, seq_len=CAP_TOKENS)

        print("[4/5] Building Transformer (DIT) ...")
        self.tr = ZImageTransformerTTNN(self.mesh_device)

        print("[5/5] Loading TTNN VAE decoder ...")
        self.vae = VaeDecoderTTNN(self.mesh_device)

        self._te_trace_id = None
        self._te_input_buf = None
        self._te_output_ref = None

        self._dit_trace_id = None
        self._lat_buf = None
        self._ts_buf = None
        self._dit_output_ref = None

        self._vae_trace_id = None
        self._vae_input_buf = None
        self._vae_output_ref = None

        print(f"Model loading: {(time.time() - t0) * 1000:.0f} ms\n")

    # ── Text encoding helpers ───────────────────────────────────────────────────

    def _tokenize(self, prompt):
        """Tokenize a prompt → [1, CAP_TOKENS] INT32 CPU tensor."""
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=CAP_TOKENS,
            return_tensors="pt",
        )["input_ids"]

    def _encode_prompt_no_trace(self, prompt):
        """Tokenize + encode without trace (used during compile phase)."""
        input_ids = self._tokenize(prompt)
        tt_input_ids = _to_device_int32(input_ids, self.mesh_device)
        tt_cap_out = self.te(tt_input_ids)
        cap_cpu = _tt_to_torch(tt_cap_out, self.mesh_device)[:CAP_TOKENS].bfloat16()
        ttnn.deallocate(tt_input_ids, force=True)
        ttnn.deallocate(tt_cap_out, force=True)
        return cap_cpu.unsqueeze(0)  # [1, CAP_TOKENS, 2560]

    def _encode_prompt(self, prompt):
        """Tokenize + encode via TE trace → [1, CAP_TOKENS, 2560] BF16 CPU tensor."""
        input_ids = self._tokenize(prompt).to(torch.int32)
        _copy_to_persistent(input_ids, self._te_input_buf, dtype=ttnn.DataType.INT32)
        ttnn.execute_trace(self.mesh_device, self._te_trace_id, cq_id=0, blocking=True)
        cap_cpu = _tt_to_torch(self._te_output_ref, self.mesh_device)[:CAP_TOKENS].bfloat16()
        return cap_cpu.unsqueeze(0)  # [1, CAP_TOKENS, 2560]

    # ── VAE decode helpers ─────────────────────────────────────────────────────

    def _decode_latents_no_trace(self, latents):
        """VAE decode without trace (used during compile phase)."""
        image_tensor = self.vae(latents)
        gc.collect()
        return self.vae_processor.postprocess(image_tensor, output_type="pil")[0]

    def _decode_latents(self, latents):
        """VAE decode via trace → PIL image."""
        preprocessed = self.vae.preprocess(latents)
        _copy_to_persistent_device(preprocessed, self._vae_input_buf)
        ttnn.deallocate(preprocessed, False)
        ttnn.execute_trace(self.mesh_device, self._vae_trace_id, cq_id=0, blocking=True)
        out = ttnn.to_torch(
            ttnn.from_device(self._vae_output_ref),
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )
        image_tensor = out[: out.shape[0] // 4].float()
        return self.vae_processor.postprocess(image_tensor, output_type="pil")[0]

    # ── Warmup: compile + capture TE, DIT & VAE traces ─────────────────────────

    def warmup(self, prompt="a cat sitting on a mat", steps=9, seed=42):
        """Compile all models, capture TE + DIT + VAE traces, run one full generation.

        After this call, forward() uses cached traces for all three models.
        """
        print("=" * 72)
        print("WARMUP: compile all programs → capture TE + DIT + VAE traces → generate")
        print("=" * 72)
        t_total = time.time()

        # ── Phase 1: Compile ALL programs (TE, DIT, VAE) BEFORE any trace capture.
        # Traces bake in device memory addresses.  Any program compiled AFTER
        # trace capture may place its binary at an address the trace uses for
        # intermediates; the next trace replay then overwrites the binary and the
        # affected model hangs.

        # 1a) Compile TE programs.
        t0 = time.time()
        cap_cpu = self._encode_prompt_no_trace(prompt)
        print(f"  TE compile: {(time.time() - t0) * 1000:.0f} ms")

        # 1b) Compile DIT programs.
        self.tr.set_cap_feats(cap_cpu)
        torch.manual_seed(seed)
        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)

        t0_step = scheduler.timesteps[0]
        t_norm_0 = max((1000.0 - float(t0_step)) / 1000.0, 1e-3)
        lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()
        ts_pt = torch.tensor([t_norm_0], dtype=torch.bfloat16)

        self._lat_buf = _to_device_bf16(lat_pt, self.mesh_device)
        self._ts_buf = _to_device_bf16(ts_pt, self.mesh_device)

        t0 = time.time()
        compile_out = self.tr._forward_impl([self._lat_buf], self._ts_buf)
        print(f"  DIT compile: {(time.time() - t0) * 1000:.0f} ms")

        # Read DIT output so we can feed it to VAE for compilation.
        dit_out_cpu = _tt_to_torch(compile_out[0], self.mesh_device)
        dit_out_cpu = dit_out_cpu.squeeze(1).unsqueeze(0).bfloat16()
        compile_latents = scheduler.step(-dit_out_cpu.float(), scheduler.timesteps[0], latents, return_dict=False)[0]

        for t in compile_out:
            ttnn.deallocate(t, force=True)

        # 1c) Compile VAE programs (first decode, pays consteval cost).
        t0 = time.time()
        _ = self._decode_latents_no_trace(compile_latents)
        print(f"  VAE compile: {(time.time() - t0) * 1000:.0f} ms")

        # ── Phase 2: Capture traces (all programs already compiled).

        # 2a) Capture TE trace.
        input_ids = self._tokenize(prompt)
        self._te_input_buf = _to_device_int32(input_ids, self.mesh_device)

        t0 = time.time()
        self._te_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._te_output_ref = self.te(self._te_input_buf)
        ttnn.end_trace_capture(self.mesh_device, self._te_trace_id, cq_id=0)
        print(f"  TE trace capture: {(time.time() - t0) * 1000:.0f} ms")

        # 2b) Capture DIT trace.
        t0 = time.time()
        self._dit_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        trace_out = self.tr._forward_impl([self._lat_buf], self._ts_buf)
        ttnn.end_trace_capture(self.mesh_device, self._dit_trace_id, cq_id=0)
        self._dit_output_ref = trace_out[0]
        print(f"  DIT trace capture: {(time.time() - t0) * 1000:.0f} ms")

        # 2c) Capture VAE trace.
        self._vae_input_buf = self.vae.preprocess(compile_latents)

        t0 = time.time()
        self._vae_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._vae_output_ref = self.vae.forward_device(self._vae_input_buf)
        ttnn.end_trace_capture(self.mesh_device, self._vae_trace_id, cq_id=0)
        print(f"  VAE trace capture: {(time.time() - t0) * 1000:.0f} ms")

        # ── Phase 3: Full warmup generation using all traces.
        torch.manual_seed(seed)
        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)

        t0 = time.time()
        cap_cpu = self._encode_prompt(prompt)
        print(f"  warmup TE (traced): {(time.time() - t0) * 1000:.0f} ms")

        _copy_to_persistent(cap_cpu, self.tr._cap_feats_buf)

        # The scheduler produces N timesteps but the last one (t=0) is a no-op
        # (dt=0, so latents += 0). Skip it to save one full DIT execution.
        active_timesteps = scheduler.timesteps[:-1]
        t0 = time.time()
        for i, t in enumerate(active_timesteps):
            t_norm = max((1000.0 - float(t)) / 1000.0, 1e-3)
            lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()
            ts_pt = torch.tensor([t_norm], dtype=torch.bfloat16)

            _copy_to_persistent(lat_pt, self._lat_buf)
            _copy_to_persistent(ts_pt, self._ts_buf)

            ttnn.execute_trace(self.mesh_device, self._dit_trace_id, cq_id=0, blocking=True)

            out = _tt_to_torch(self._dit_output_ref, self.mesh_device)
            out = out.squeeze(1).unsqueeze(0).bfloat16()
            latents = scheduler.step(-out.float(), t, latents, return_dict=False)[0]
        print(f"  warmup denoising ({len(active_timesteps)} steps): {(time.time() - t0) * 1000:.0f} ms")

        t0 = time.time()
        image = self._decode_latents(latents)
        print(f"  VAE decode (traced): {(time.time() - t0) * 1000:.0f} ms")

        print(f"  WARMUP TOTAL: {(time.time() - t_total) * 1000:.0f} ms")
        _dram_stats(self.mesh_device, "after warmup")
        print("=" * 72)
        print("TE + DIT + VAE traces captured. Ready for fast generation.\n")
        return image

    # ── Fast generation using cached trace ────────────────────────────────────

    def forward(self, prompt, steps=9, seed=42):
        """Generate an image using cached TE + DIT + VAE traces.

        Must call warmup() first. Returns a PIL Image.
        """
        if self._dit_trace_id is None:
            raise RuntimeError("Call warmup() before forward().")

        t_total = time.time()
        _dram_stats(self.mesh_device, "before generate")

        # 1) Text encoding via TE trace.
        t0 = time.time()
        cap_cpu = self._encode_prompt(prompt)
        te_ms = (time.time() - t0) * 1000
        print(f"  text encoding (traced): {te_ms:.0f} ms")

        # 2) Copy new caption features to the DIT's persistent buffer.
        #    MUST NOT call set_cap_feats() — that allocates a new tensor and
        #    breaks the trace which baked in the old address.
        t0 = time.time()
        _copy_to_persistent(cap_cpu, self.tr._cap_feats_buf)
        print(f"  cap_feats copy: {(time.time() - t0) * 1000:.0f} ms")

        # 3) Init noise + scheduler.
        torch.manual_seed(seed)
        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)

        # 4) Denoising loop — DIT via trace replay.
        # Skip last timestep (t=0, dt=0 → no-op). 8 DIT forwards with 9-step schedule.
        active_timesteps = scheduler.timesteps[:-1]
        num_steps = len(active_timesteps)
        step_times = []
        for i, t in enumerate(active_timesteps):
            t0 = time.time()

            t_norm = max((1000.0 - float(t)) / 1000.0, 1e-3)
            lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()  # [16,1,64,64]
            ts_pt = torch.tensor([t_norm], dtype=torch.bfloat16)  # [1]

            _copy_to_persistent(lat_pt, self._lat_buf)
            _copy_to_persistent(ts_pt, self._ts_buf)

            ttnn.execute_trace(self.mesh_device, self._dit_trace_id, cq_id=0, blocking=True)

            out = _tt_to_torch(self._dit_output_ref, self.mesh_device)
            out = out.squeeze(1).unsqueeze(0).bfloat16()
            latents = scheduler.step(-out.float(), t, latents, return_dict=False)[0]

            elapsed = (time.time() - t0) * 1000
            step_times.append(elapsed)
            print(f"  step {i + 1}/{num_steps}: {elapsed:.0f} ms")

        # 5) VAE decode via trace.
        t0 = time.time()
        image = self._decode_latents(latents)
        vae_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - t_total) * 1000
        steady = step_times[1:] if len(step_times) > 1 else step_times
        avg_steady = sum(steady) / len(steady) if steady else 0
        print(
            f"  VAE (traced): {vae_ms:.0f} ms  |  total: {total_ms:.0f} ms  "
            f"|  steady-state step: {avg_steady:.0f} ms avg"
        )
        _dram_stats(self.mesh_device, "after generate")
        return image
