# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The whole dots.ocr model in TP4: vision tower + text prefill + paged decode.

This stitches the three TP4 pieces into a single image->text model:

  * vision tower  : ``TTNNDotsOCRVisionTowerTP4BH`` (Blackhole TP4, hardware-swept
                    kernels for the S=11264 / grid 88x128 vision bucket). Emits the
                    merged image embeddings ``[1,1,2816,H]`` column-sharded on H.
  * text prefill  : ``DotsOCRPrefillModelTP4`` (replicated-hidden Megatron TP4).
                    Fills the paged KV cache and emits the first token's logits.
  * text decode   : the same model's paged ``decode_with_head`` reading/extending
                    the KV cache one token at a time.

Design note -- the vision tower emits a *column-sharded* hidden stream while the
text decoder consumes a *replicated, full-width* hidden stream. Rather than
reconcile the two shardings on device, the text embedding + vision scatter-merge
is done on host (torch): we gather the vision shards back to a full-H torch
tensor, drop them into the image-placeholder positions of the text embedding, and
feed the merged sequence to the text model replicated. This mirrors how every
existing ``dots_ocr_tp4`` test feeds the text path and keeps the replicated-hidden
prefill/decode untouched.
"""

import os
import time

import torch
import ttnn

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, to_replicated
from models.experimental.dots_ocr_tp4.tt.graphs import (
    DotsOCRDecodeGraphTP4,
    DotsOCRPrefillGraphTP4,
    DotsOCRVisionGraphTP4,
)
from models.experimental.dots_ocr_tp4.tt.kv_cache import create_paged_kv_cache
from models.experimental.dots_ocr_tp4.tt.model import DotsOCRPrefillModelTP4

# Vision tower lives in tt_symbiote; the text TP4 rebuild already reuses its rope
# and paged-cache, so this is consistent with the rest of dots_ocr_tp4.
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsOCRVisionTowerTP4BH
from models.experimental.tt_symbiote.utils.device_management import set_device


# The vision TP4 BH kernels are hardware-swept for exactly this grid.
VISION_GRID_THW = [1, 88, 128]  # (t, h_patches, w_patches) -> 11264 patches
VISION_MERGED_TOKENS = 2816  # 11264 // spatial_merge_size(2)^2
DEFAULT_IMAGE_TOKEN_ID = 151665  # "<|imgpad|>"


def _raw_ttnn(t):
    """Unwrap a ``TorchTTNNTensor`` to the underlying ``ttnn.Tensor`` (if wrapped)."""
    inner = getattr(t, "ttnn_tensor", None)
    return inner if inner is not None else t


def _pos_tensor(mesh_device, pos: int):
    """Replicated int32 ``[1]`` current-position tensor for paged decode."""
    return ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
    )


class DotsOCRModelTP4:
    """Full dots.ocr (vision + text) in the TP4 configuration."""

    def __init__(
        self,
        mesh_device,
        config: DotsOCRConfig,
        vision_tower,
        text_model: DotsOCRPrefillModelTP4,
        embed_tokens,
        image_token_id: int = DEFAULT_IMAGE_TOKEN_ID,
        eos_ids=None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.vision_tower = vision_tower
        self.text_model = text_model
        self.embed_tokens = embed_tokens  # host (CPU) nn.Embedding
        self.image_token_id = int(image_token_id)
        self.eos_ids = set(eos_ids or [])
        # Device timings (seconds) from the most recent generate(); see generate().
        self.last_timings = {}
        # Persistent paged KV cache (reused + reset() across warmup/generate so the
        # captured trace's cache-buffer references stay valid on replay) + the
        # @trace_enabled graphs (the framework trace boundaries).
        self.cache = None
        self.graph_prefill = None
        self.graph_decode = None
        self.graph_vision = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_hf(cls, mesh_device, hf_model, weight_dtype=ttnn.bfloat16, build_vision=True):
        """Build the integrated TP4 model from a loaded HF dots.ocr model.

        ``build_vision=False`` skips the vision-tower build (text-only runs, e.g.
        the traced-decode benchmark) to save setup time and device DRAM.

        ``hf_model`` is the in-memory ``AutoModelForCausalLM`` (text + vision).
        The TP4 attention/MLP weight loaders build *new* device tensors and do
        not mutate the HF layers in place, so ``hf_model.model.embed_tokens`` is
        kept on host and reused for token embedding during generation.
        """
        config = DotsOCRConfig.from_hf(hf_model.config)
        text_root = hf_model.model

        # --- Text decoder body + final norm + LM head (replicated-hidden TP4) ---
        text_model = DotsOCRPrefillModelTP4.from_torch(
            mesh_device,
            config,
            text_root.layers,
            torch_norm=text_root.norm,
            torch_lm_head=hf_model.lm_head,
            weight_dtype=weight_dtype,
        )

        # --- Vision tower (Blackhole TP4) ---
        vision_tower = None
        if build_vision:
            vision_tower = TTNNDotsOCRVisionTowerTP4BH.from_torch(hf_model.vision_tower, hf_model.config)
            set_device(vision_tower, mesh_device, register_forward_hook=False, dump_visualization=False)
            vision_tower.preprocess_weights()
            vision_tower.move_weights_to_device()

        eos = hf_model.config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])

        m = cls(
            mesh_device,
            config,
            vision_tower,
            text_model,
            embed_tokens=text_root.embed_tokens,
            image_token_id=getattr(hf_model.config, "image_token_id", DEFAULT_IMAGE_TOKEN_ID),
            eos_ids=eos_ids,
        )

        # @trace_enabled graph boundaries over the (already on-device) text model.
        # Invoking graph(...) routes through the TTNNModule framework; under
        # TT_SYMBIOTE_RUN_MODE=TRACED it captures/replays, else it runs eager.
        m.graph_prefill = DotsOCRPrefillGraphTP4(text_model)
        m.graph_decode = DotsOCRDecodeGraphTP4(text_model)
        graphs = [m.graph_prefill, m.graph_decode]
        # Vision graph: prebuild the fixed-bucket RoPE once (outside any trace) so
        # the captured trunk reads stable rope buffers (no per-call rope.build).
        # OPT-IN (off by default): the BH TP4 vision trunk does host<->device
        # transfers inside its forward (per-call attention-mask / constant builds)
        # which trace capture rejects ("Writes/Reads not supported during trace
        # capture"). Tracing it cleanly needs those hoisted out of the trunk
        # (build masks outside, as the production pipeline does) -- a vision_tp4_bh
        # change, not a wrapper. Until then vision runs eager (prefill+decode traced).
        if vision_tower is not None and os.environ.get("DOTS_OCR_TP4_VISION_TRACE"):
            from models.experimental.tt_symbiote.modules.vision_tp4_bh import rot_mats_l1

            grid = torch.tensor([VISION_GRID_THW], dtype=torch.int32)
            seq_len = VISION_GRID_THW[0] * VISION_GRID_THW[1] * VISION_GRID_THW[2]
            rot_mats, cu_seqlens = vision_tower.rope.build(grid, seq_len)
            rot_mats = rot_mats_l1(rot_mats)
            m.graph_vision = DotsOCRVisionGraphTP4(vision_tower, rot_mats, cu_seqlens)
            graphs.append(m.graph_vision)
        for g in graphs:
            set_device(g, mesh_device, register_forward_hook=False, dump_visualization=False)
        # Persistent paged cache (reset() between runs; never recreated, so trace
        # capture/replay keep referencing the same cache buffers).
        m.cache = create_paged_kv_cache(config, mesh_device, batch_size=1)
        TracedRun.configure(device=mesh_device, cq_id=0)
        return m

    # --------------------------------------------------------------- vision
    def _vision_embeds_host(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, use_trace: bool = False
    ) -> torch.Tensor:
        """Run the vision tower and return merged image embeds as host ``[N, H]`` bf16.

        The tower output is column-sharded on H across the 4 chips; gather the
        shards back to the full hidden dim on host. With ``use_trace``, ``patch_embed``
        runs eager (variable per image) and the fixed-bucket trunk runs through the
        traced ``graph_vision``.
        """
        pv = pixel_values.to(torch.bfloat16)
        if use_trace and self.graph_vision is not None:
            x = self.vision_tower.patch_embed(pv, image_grid_thw)  # outside the trace
            out = self.graph_vision(x)
        else:
            out = self.vision_tower.forward(pv, image_grid_thw)
        ttnn.synchronize_device(self.mesh_device)
        out = _raw_ttnn(out)
        if self.mesh_device.get_num_devices() > 1:
            v = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        else:
            v = ttnn.to_torch(out)
        # [1, 1, N, H] -> [N, H]
        v = v.reshape(int(v.shape[-2]), int(v.shape[-1]))
        return v.to(torch.bfloat16)

    # ---------------------------------------------------------------- merge
    def _merge_embeds_host(self, input_ids: torch.Tensor, vision_embeds: torch.Tensor) -> torch.Tensor:
        """Text-embed ``input_ids`` and scatter ``vision_embeds`` into image slots.

        Returns ``[1, L, H]`` bf16 on host. Image-placeholder positions
        (``input_ids == image_token_id``) are replaced, in order, by the merged
        vision tokens.
        """
        embeds = self.embed_tokens(input_ids).to(torch.bfloat16)  # [1, L, H]
        mask = input_ids[0] == self.image_token_id  # [L] bool
        n_img = int(mask.sum().item())
        if n_img == 0:
            return embeds
        if vision_embeds.shape[0] < n_img:
            raise ValueError(
                f"vision produced {vision_embeds.shape[0]} tokens but prompt has {n_img} image placeholders"
            )
        embeds = embeds.clone()
        embeds[0, mask] = vision_embeds[:n_img].to(embeds.dtype)
        return embeds

    # ------------------------------------------------------------- generate
    def _embeds_to_replicated(self, embeds: torch.Tensor):
        """Tile-pad ``[1, L, H]`` on the seq dim and push it to the mesh replicated.

        Returns ``(x_tt, L)``. Causal attention makes the right-pad tail "future"
        tokens that don't affect earlier positions, so the head can be read at the
        real last position ``L-1``.
        """
        L = int(embeds.shape[1])
        H = int(embeds.shape[2])
        s_pad = ((L + 31) // 32) * 32
        if s_pad > L:
            pad = torch.zeros(1, s_pad - L, H, dtype=embeds.dtype)
            embeds = torch.cat([embeds, pad], dim=1)
        x_tt = to_replicated(embeds.to(torch.bfloat16), self.mesh_device, dtype=ttnn.bfloat16)
        return x_tt, L

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        max_new_tokens: int = 128,
        stop_on_eos: bool = True,
        use_trace: bool = False,
    ) -> list[int]:
        """Greedy generation: vision -> merge -> prefill (fill KV) -> paged decode.

        ``input_ids`` is a torch long ``[1, L]``. With ``pixel_values`` provided,
        the image placeholders in ``input_ids`` are filled with vision embeddings.
        Returns the list of newly generated token ids.

        ``use_trace`` runs the decode step as a captured ttnn trace replayed per
        token (removes per-op host dispatch, the dominant cost at M=1); vision and
        prefill always run eager. Requires the device to be opened with a non-zero
        ``trace_region_size``.
        """
        if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError(f"expected input_ids [1, L], got {tuple(input_ids.shape)}")

        # Reuse the persistent cache (reset, not recreated) so traced graphs that
        # captured ops on its buffers replay correctly.
        cache = self.cache
        cache.reset()

        # Device timing is measured as wall-clock bracketed by synchronize_device,
        # so each interval reflects when the device actually finished that work.
        ttnn.synchronize_device(self.mesh_device)
        t_start = time.perf_counter()

        # --- Build the (merged) input embedding sequence on host. ---
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required when pixel_values is provided")
            if self.vision_tower is None:
                raise RuntimeError("pixel_values given but model was built with build_vision=False")
            # _vision_embeds_host syncs the device internally, so t_vision below
            # captures the vision tower's device time.
            vision_embeds = self._vision_embeds_host(pixel_values, image_grid_thw, use_trace=use_trace)
            t_vision = time.perf_counter()
            embeds = self._merge_embeds_host(input_ids, vision_embeds)
        else:
            embeds = self.embed_tokens(input_ids).to(torch.bfloat16)
            t_vision = time.perf_counter()

        # --- Prefill: fill the KV cache, read the first token at the real last pos. ---
        x_tt, L0 = self._embeds_to_replicated(embeds)
        if use_trace:
            # Through the @trace_enabled graph (framework trace under TRACED mode).
            lv, li = self.graph_prefill(x_tt, past_key_value=cache, token_index=L0 - 1)
            first = self._combine_dist(lv, li)
        else:
            _, tok = self.text_model.prefill_with_head(x_tt, cache, token_index=L0 - 1, return_token=True)
            first = int(tok.flatten()[0])
        ttnn.synchronize_device(self.mesh_device)
        t_prefill = time.perf_counter()

        # --- Decode: eager (direct forward) or graph (framework trace). ---
        out_ids = [first]
        if use_trace:
            decode_s, n_decode = self._graph_decode_loop(cache, first, L0, max_new_tokens, stop_on_eos, out_ids)
        else:
            decode_s, n_decode = self._eager_decode_loop(cache, first, L0, max_new_tokens, stop_on_eos, out_ids)

        self.last_timings = {
            "vision_s": t_vision - t_start,
            "prefill_s": t_prefill - t_vision,
            "vision_prefill_s": t_prefill - t_start,
            "decode_s": decode_s,
            "decode_tokens": n_decode,
            "decode_ms_per_token": (decode_s / n_decode * 1000.0) if n_decode else 0.0,
            "decode_tok_per_s": (n_decode / decode_s) if decode_s > 0 and n_decode else 0.0,
            "traced": use_trace,
        }
        return out_ids

    # ---------------------------------------------------------- decode loops
    def _eager_decode_loop(self, cache, first, L0, max_new_tokens, stop_on_eos, out_ids):
        """One token at a time, fresh ops each step. Returns (decode_s, n_tokens)."""
        md = self.mesh_device
        prev, pos = first, L0  # the freshly generated `first` sits at position L0
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if stop_on_eos and prev in self.eos_ids:
                break
            emb = self.embed_tokens(torch.tensor([[prev]], dtype=torch.long)).to(torch.bfloat16)  # [1,1,H]
            x_tt = to_replicated(emb, md, dtype=ttnn.bfloat16)
            cur_pos = _pos_tensor(md, pos)
            _, tok = self.text_model.decode_with_head(x_tt, cache, cur_pos, return_token=True)
            ttnn.synchronize_device(md)
            nxt = int(tok.flatten()[0])
            out_ids.append(nxt)
            prev, pos = nxt, pos + 1
        ttnn.synchronize_device(md)
        return time.perf_counter() - t0, len(out_ids) - 1

    def _combine_dist(self, local_val, local_idx) -> int:
        """Gather the nd per-chip (value, local_index) candidates to host and pick
        the global argmax token = winner_chip * v_shard + local_index[winner].
        Accepts ttnn.Tensors or framework-wrapped TorchTTNNTensors (graph outputs)."""
        md = self.mesh_device
        local_val = _raw_ttnn(local_val)
        local_idx = _raw_ttnn(local_idx)
        if md.get_num_devices() > 1:
            vals = ttnn.to_torch(local_val, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float().flatten()
            idxs = ttnn.to_torch(local_idx, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).flatten()
        else:
            vals = ttnn.to_torch(local_val).float().flatten()
            idxs = ttnn.to_torch(local_idx).flatten()
        winner = int(vals.argmax())
        return winner * int(self.text_model.head.v_shard) + int(idxs[winner])

    def _graph_decode_loop(self, cache, first, L0, max_new_tokens, stop_on_eos, out_ids):
        """Decode via the @trace_enabled graph. Under TRACED mode the framework
        warms (step 0), captures (step 1) and replays (step 2+) automatically, and
        ttnn.copy's the per-token embedding + position into its own persistent trace
        buffers. Times the steady-state region (skips the first two warm/capture
        steps); after warmup() all steps are replays, so the skip only drops
        already-warm replays."""
        md = self.mesh_device
        prev, pos = first, L0
        t_steady = None
        n_steady = 0
        t_write = t_exec = t_read = 0.0
        for i in range(max_new_tokens - 1):
            if stop_on_eos and prev in self.eos_ids:
                break
            a = time.perf_counter()
            emb = self.embed_tokens(torch.tensor([[prev]], dtype=torch.long)).to(torch.bfloat16)  # [1,1,H]
            x_tt = to_replicated(emb, md, dtype=ttnn.bfloat16)
            cp = _pos_tensor(md, pos)
            b = time.perf_counter()
            lv, li = self.graph_decode(x_tt, cp, past_key_value=cache)
            ttnn.synchronize_device(md)
            c = time.perf_counter()
            nxt = self._combine_dist(lv, li)
            d = time.perf_counter()
            out_ids.append(nxt)
            prev, pos = nxt, pos + 1
            if i == 1:
                ttnn.synchronize_device(md)
                t_steady = time.perf_counter()
            elif i >= 2:
                t_write += b - a
                t_exec += c - b
                t_read += d - c
                n_steady += 1
        decode_s = (time.perf_counter() - t_steady) if (t_steady is not None and n_steady) else 0.0
        if n_steady:
            self.last_timings_breakdown = {
                "write_ms_per_token": t_write / n_steady * 1000.0,
                "exec_ms_per_token": t_exec / n_steady * 1000.0,
                "read_ms_per_token": t_read / n_steady * 1000.0,
            }
        return decode_s, n_steady

    # ------------------------------------------------------------- warmup
    def warmup(self, input_ids: torch.Tensor, pixel_values: torch.Tensor = None, image_grid_thw: torch.Tensor = None):
        """Prime JIT + capture the prefill/decode (and, if ``pixel_values`` given,
        vision) traces, mirroring the production pipeline's warmup. Call with the
        SAME shapes as the timed generate so the captured trace keys match (the
        framework keys on input shape). Text warmup runs text-only (the prefill
        embedding shape only depends on prompt length); the real generate replays
        with the vision-merged embeddings copied in."""
        self.generate(input_ids, max_new_tokens=2, stop_on_eos=False, use_trace=True)
        self.cache.reset()
        TracedRun.release_all()
        self.generate(input_ids, max_new_tokens=4, stop_on_eos=False, use_trace=True)
        self.cache.reset()
        # Vision trunk trace (fixed 88x128 bucket): patch_embed eager, then warm
        # (encounter 1) + capture (encounter 2) the trunk so the real generate
        # replays. Rebuild ``x`` each pass -- the trunk consumes/deallocates its
        # input, so the same patch-embed tensor can't be fed twice.
        if self.graph_vision is not None and pixel_values is not None:
            pv = pixel_values.to(torch.bfloat16)
            for _ in range(2):
                x = self.vision_tower.patch_embed(pv, image_grid_thw)
                self.graph_vision(x)
                ttnn.synchronize_device(self.mesh_device)

    def release(self):
        """Release captured traces."""
        TracedRun.release_all()
