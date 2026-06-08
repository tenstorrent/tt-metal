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

import time

import torch
import ttnn

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, to_replicated
from models.experimental.dots_ocr_tp4.tt.kv_cache import create_paged_kv_cache
from models.experimental.dots_ocr_tp4.tt.model import DotsOCRPrefillModelTP4

# Vision tower lives in tt_symbiote; the text TP4 rebuild already reuses its rope
# and paged-cache, so this is consistent with the rest of dots_ocr_tp4.
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

    # ------------------------------------------------------------------ build
    @classmethod
    def from_hf(cls, mesh_device, hf_model, weight_dtype=ttnn.bfloat16):
        """Build the integrated TP4 model from a loaded HF dots.ocr model.

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
        vision_tower = TTNNDotsOCRVisionTowerTP4BH.from_torch(hf_model.vision_tower, hf_model.config)
        set_device(vision_tower, mesh_device, register_forward_hook=False, dump_visualization=False)
        vision_tower.preprocess_weights()
        vision_tower.move_weights_to_device()

        eos = hf_model.config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])

        return cls(
            mesh_device,
            config,
            vision_tower,
            text_model,
            embed_tokens=text_root.embed_tokens,
            image_token_id=getattr(hf_model.config, "image_token_id", DEFAULT_IMAGE_TOKEN_ID),
            eos_ids=eos_ids,
        )

    # --------------------------------------------------------------- vision
    def _vision_embeds_host(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Run the vision tower and return merged image embeds as host ``[N, H]`` bf16.

        The tower output is column-sharded on H across the 4 chips; gather the
        shards back to the full hidden dim on host.
        """
        out = self.vision_tower.forward(pixel_values.to(torch.bfloat16), image_grid_thw)
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
    ) -> list[int]:
        """Greedy generation: vision -> merge -> prefill (fill KV) -> paged decode.

        ``input_ids`` is a torch long ``[1, L]``. With ``pixel_values`` provided,
        the image placeholders in ``input_ids`` are filled with vision embeddings.
        Returns the list of newly generated token ids.
        """
        if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError(f"expected input_ids [1, L], got {tuple(input_ids.shape)}")

        cache = create_paged_kv_cache(self.config, self.mesh_device, batch_size=1)

        # Device timing is measured as wall-clock bracketed by synchronize_device,
        # so each interval reflects when the device actually finished that work.
        ttnn.synchronize_device(self.mesh_device)
        t_start = time.perf_counter()

        # --- Build the (merged) input embedding sequence on host. ---
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required when pixel_values is provided")
            # _vision_embeds_host syncs the device internally, so t_vision below
            # captures the vision tower's device time.
            vision_embeds = self._vision_embeds_host(pixel_values, image_grid_thw)
            t_vision = time.perf_counter()
            embeds = self._merge_embeds_host(input_ids, vision_embeds)
        else:
            embeds = self.embed_tokens(input_ids).to(torch.bfloat16)
            t_vision = time.perf_counter()

        # --- Prefill: fill the KV cache, read the first token at the real last pos. ---
        x_tt, L0 = self._embeds_to_replicated(embeds)
        _, tok = self.text_model.prefill_with_head(x_tt, cache, token_index=L0 - 1, return_token=True)
        ttnn.synchronize_device(self.mesh_device)
        t_prefill = time.perf_counter()
        first = int(tok.flatten()[0])

        # --- Decode: one token at a time, reading/extending the paged cache. ---
        out_ids = [first]
        prev = first
        pos = L0  # the freshly generated token sits at position L0
        for _ in range(max_new_tokens - 1):
            if stop_on_eos and prev in self.eos_ids:
                break
            emb = self.embed_tokens(torch.tensor([[prev]], dtype=torch.long)).to(torch.bfloat16)  # [1,1,H]
            x_tt = to_replicated(emb, self.mesh_device, dtype=ttnn.bfloat16)
            cur_pos = _pos_tensor(self.mesh_device, pos)
            _, tok = self.text_model.decode_with_head(x_tt, cache, cur_pos, return_token=True)
            ttnn.synchronize_device(self.mesh_device)
            nxt = int(tok.flatten()[0])
            out_ids.append(nxt)
            prev = nxt
            pos += 1
        ttnn.synchronize_device(self.mesh_device)
        t_end = time.perf_counter()

        n_decode = len(out_ids) - 1  # tokens from the decode loop (first came from prefill)
        decode_s = t_end - t_prefill
        self.last_timings = {
            "vision_s": t_vision - t_start,
            "prefill_s": t_prefill - t_vision,
            "vision_prefill_s": t_prefill - t_start,
            "decode_s": decode_s,
            "decode_tokens": n_decode,
            "decode_ms_per_token": (decode_s / n_decode * 1000.0) if n_decode else 0.0,
            "decode_tok_per_s": (n_decode / decode_s) if decode_s > 0 and n_decode else 0.0,
        }
        return out_ids
