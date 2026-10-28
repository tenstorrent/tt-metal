# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""d1-3B on one Wormhole chip: lfm25_vl's hybrid LFM2 model, prefill only, read at the answer slot.

A prompt's embeddings are looked up on host (where image features are spliced in), right-padded with zero rows to
the smallest bucket that holds them, and copied into that bucket's persistent input. Every operator is causal, so
the padding never reaches the last real token. One trace per bucket runs the 30 layers, selects the last real
row with a one-hot matmul and applies the final norm; the host reads that one row back.

The LM head is tied to the embedding, so an option's logit is a dot product of that row with the option's
embedding row: the host does it in fp32 for the handful of option tokens instead of a 128K-vocab matmul.
"""

import time
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn
from models.demos.multimodal.d1_3b.tt.model_config import PREFILL_BUCKETS, D1ModelArgs
from models.demos.multimodal.lfm25_vl.tt.e2e_model import TtLfm25VlModel
from models.tt_transformers.tt.common import Mode

TRACE_REGION_SIZE = 64 * 1024 * 1024
L1_SMALL_SIZE = 24576
# The first pass compiles every program, the second takes the steady-state path.
WARM_PASSES = 2


@dataclass
class _Bucket:
    size: int
    embeds: ttnn.Tensor
    sel: ttnn.Tensor
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    trace_id: int = None
    out: ttnn.Tensor = None


class TtD1Model:
    def __init__(self, mesh_device, dtype=ttnn.bfloat16, n_layers=None, buckets=PREFILL_BUCKETS):
        self.mesh_device = mesh_device
        self.args = D1ModelArgs(mesh_device)
        self.args.prefill_only = True
        if n_layers is not None:
            self.args.n_layers = n_layers
            self.args.layer_types = self.args.layer_types[:n_layers]
        state_dict = self.args.load_state_dict()
        self.embedding = state_dict["tok_embeddings.weight"].to(torch.bfloat16).clone()
        self.model = TtLfm25VlModel(
            args=self.args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=self.args.weight_cache_path(dtype),
        )
        del state_dict
        self.sizes = tuple(b for b in buckets if b <= self.args.max_seq_len)
        self.buckets = {}
        self.traced = False
        self.select_cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    # ------------------------------------------------------------------ inputs

    def embed(self, input_ids, vision_embeddings=None):
        """Host embeddings [n, dim] bf16, with `vision_embeddings` [n_image_tokens, dim] at the image tokens."""
        ids = torch.tensor(input_ids, dtype=torch.long)
        x = self.embedding[ids]
        if vision_embeddings is not None:
            slots = ids == self.args.image_token_id
            assert int(slots.sum()) == vision_embeddings.shape[0], "image tokens and image features differ in number"
            x[slots] = vision_embeddings.to(torch.bfloat16)
        return x

    def vision_embeddings(self, pixel_values, spatial_shapes=None, pixel_attention_mask=None, patch_multiple=128):
        """Host image-token embeddings [n_image_tokens, dim] from the SigLIP2 tower + projector (reused from lfm25_vl).

        The device encoder needs the patch axis a multiple of `patch_multiple`; the HF processor/runner may hand
        back a trimmed count. Pad the patches with zeros and mark them invalid so the attention mask and the
        per-image unpad (which both read `pixel_attention_mask`) ignore them.
        """
        n = pixel_values.shape[1]
        pad = (-n) % patch_multiple
        if pad:
            pixel_values = torch.nn.functional.pad(pixel_values, (0, 0, 0, pad))
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(pixel_values.shape[0], n, dtype=torch.long)
            pixel_attention_mask = torch.nn.functional.pad(pixel_attention_mask, (0, pad))
        return self.model.encode_vision_embeddings_from_pixels(
            pixel_values, spatial_shapes=spatial_shapes, pixel_attention_mask=pixel_attention_mask
        )

    def _bucket_for(self, num_tokens):
        size = next((s for s in self.sizes if s >= num_tokens), None)
        if size is None:
            raise ValueError(f"{num_tokens} tokens exceed the largest prefill bucket ({self.sizes[-1]})")
        if size not in self.buckets:
            assert not self.traced, "every bucket is allocated before the first trace capture"
            self.buckets[size] = self._alloc_bucket(size)
        return self.buckets[size]

    def _alloc_bucket(self, size):
        def upload(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        rope = self.model.rope_setup
        return _Bucket(
            size=size,
            embeds=upload(torch.zeros(1, 1, size, self.args.dim)),
            sel=upload(torch.zeros(1, 1, 1, size)),
            cos=rope.cos_matrix_prefill[:, :, :size, :],
            sin=rope.sin_matrix_prefill[:, :, :size, :],
        )

    def _stage(self, bucket, embeds):
        n = embeds.shape[0]
        x = torch.zeros(1, 1, bucket.size, self.args.dim, dtype=torch.bfloat16)
        x[0, 0, :n] = embeds
        sel = torch.zeros(1, 1, 1, bucket.size, dtype=torch.bfloat16)
        sel[0, 0, 0, n - 1] = 1.0
        for host, device in ((x, bucket.embeds), (sel, bucket.sel)):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    host,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                ),
                device,
            )

    # ----------------------------------------------------------------- forward

    def _body(self, bucket):
        """30 layers, last-row select and final norm. Returns [1, 1, 32, dim]; row 0 is the answer slot."""
        # The first layer frees its input, and bucket.embeds must outlive the pass.
        x = ttnn.clone(bucket.embeds, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rot = [bucket.cos, bucket.sin]
        hidden = self.model.ttnn_prefill_forward(x, rot_mats_global=rot, rot_mats_local=rot, user_id=0)
        row = ttnn.matmul(bucket.sel, hidden, compute_kernel_config=self.select_cfg)
        ttnn.deallocate(hidden)
        return self.model.norm(
            row, mode=Mode.PREFILL, norm_config=self.args.get_norm_config("lm_head", Mode.PREFILL, None)
        )

    def capture_traces(self):
        """Warm every bucket, then record one trace per bucket. Nothing may be allocated after the first capture."""
        start = time.perf_counter()
        for size in self.sizes:
            bucket = self._bucket_for(size)
            self._stage(bucket, torch.zeros(1, self.args.dim, dtype=torch.bfloat16))
            for _ in range(WARM_PASSES):
                ttnn.deallocate(self._body(bucket))
            ttnn.synchronize_device(self.mesh_device)
        logger.info(f"{len(self.sizes)} buckets warmed in {time.perf_counter() - start:.1f}s")
        start = time.perf_counter()
        for size in self.sizes:
            bucket = self.buckets[size]
            bucket.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            bucket.out = self._body(bucket)
            ttnn.end_trace_capture(self.mesh_device, bucket.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.traced = True
        logger.info(f"{len(self.sizes)} bucket traces captured in {time.perf_counter() - start:.1f}s")

    def release_traces(self):
        for bucket in self.buckets.values():
            if bucket.trace_id is not None:
                ttnn.release_trace(self.mesh_device, bucket.trace_id)
                bucket.trace_id = None
        self.traced = False

    def hidden(self, input_ids, vision_embeddings=None):
        """Final-norm hidden state [dim] fp32 at the last token of `input_ids`."""
        bucket = self._bucket_for(len(input_ids))
        self._stage(bucket, self.embed(input_ids, vision_embeddings))
        if bucket.trace_id is not None:
            ttnn.execute_trace(self.mesh_device, bucket.trace_id, cq_id=0, blocking=False)
            out = ttnn.to_torch(ttnn.get_device_tensors(bucket.out)[0])
        else:
            out_t = self._body(bucket)
            out = ttnn.to_torch(ttnn.get_device_tensors(out_t)[0])
            ttnn.deallocate(out_t)
        return out.reshape(-1, out.shape[-1])[0, : self.args.dim].float()

    def option_logits(self, hidden, token_ids):
        """{token id: logit} through the tied LM head, fp32 on host."""
        rows = self.embedding[torch.tensor(token_ids, dtype=torch.long)].float()
        return dict(zip(token_ids, (rows @ hidden).tolist()))

    def logits(self, hidden, chunk=16384):
        """Full-vocab logits [vocab] fp32 on host, for accuracy tests."""
        return torch.cat([self.embedding[i : i + chunk].float() @ hidden for i in range(0, len(self.embedding), chunk)])
