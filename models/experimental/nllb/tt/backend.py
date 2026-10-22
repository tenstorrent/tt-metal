# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Configuration-driven TTNN NLLB inference; see README.md for supported scope."""

import math
from pathlib import Path
import numpy as np
import torch
import ttnn

from models.experimental.nllb.tt.nllb_validation import (
    validate_config,
    validate_checkpoint,
    validate_inputs,
    token_array,
    integer,
    validate_checkpoint_config,
    load_checkpoint,
)


DEVICE_OPTIONS = {"trace_region_size": 64 * 1024 * 1024}

# Architecture signature of distilled 600M, independent of checkpoint path/name.
# Validation still checks the supplied checkpoint against its configuration.
_DISTILLED_600M = dict(
    d_model=1024,
    vocab_size=256206,
    encoder_layers=12,
    decoder_layers=12,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
    max_position_embeddings=1024,
)


def create_backend(weights_path, config, device, *, precision="bf16"):
    return Backend(weights_path, config, device, precision=precision)


class Backend:
    def __init__(self, weights_path, config, device, *, precision="bf16"):
        if precision not in ("bf16", "bfp8_b"):
            raise ValueError("Supported precisions: bf16 and bfp8_b with BF16 activations")
        # Explicit test/control selection also works through the unchanged factory.
        self.generation_projection = Path(__file__).with_name("generation_projection.txt").read_text().strip()
        if self.generation_projection not in ("full", "last"):
            raise ValueError("generation_projection.txt must contain full or last")
        config = validate_config(config)
        validate_checkpoint_config(weights_path, config)
        if device is None:
            raise ValueError("An open caller-owned TTNN device is required")
        self.matrix_dtype = ttnn.bfloat8_b if precision == "bfp8_b" else ttnn.bfloat16
        self.precision_policy = {
            "mode": precision,
            "weights": precision,
            "activations": "bf16",
            "accumulation": "FP32 destination accumulation, HiFi4, approximate math disabled",
            "exceptions": (
                []
                if precision == "bf16"
                else [
                    "BF16 row-major token embedding for embedding operator compatibility",
                    "BF16 normalization weights and all biases",
                    "BF16 activations, masks, attention and linear outputs to limit accumulated error",
                ]
            ),
        }
        self.bf16_matrix_exceptions = frozenset()
        if precision == "bfp8_b" and all(config[key] == value for key, value in _DISTILLED_600M.items()):
            self.bf16_matrix_exceptions = frozenset(
                f"model.{side}.layers.{layer}.{attention}.{projection}.weight"
                for side, attentions in (("encoder", ("self_attn",)), ("decoder", ("self_attn", "encoder_attn")))
                for layer in range(config[side + "_layers"])
                for attention in attentions
                for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
            )
            self.bf16_matrix_exceptions |= frozenset(
                f"model.{side}.layers.{layer}.{linear}.weight"
                for side in ("encoder", "decoder")
                for layer in range(config[side + "_layers"])
                for linear in ("fc1", "fc2")
            )
            self.precision_policy["exceptions"].append(
                "600M only: BF16 checkpoint weights for all transformer attention and FFN matrices; LM head remains BFP8_B on every request"
            )
        self.trace_decoder_enabled = True
        self._trace_failures = []
        self.config = dict(config)
        self.device = device
        self.dim = int(config["d_model"])
        self.pad = int(config["pad_token_id"])
        self.vocab = int(config["vocab_size"])
        self.scale = math.sqrt(self.dim) if config.get("scale_embedding", True) else 1.0
        if config.get("activation_function", "relu") != "relu":
            raise ValueError("Only ReLU model configuration supported")
        self.kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.decoder_sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )
        print("backend: loading official CPU state dictionary", flush=True)
        weights = load_checkpoint(weights_path)
        validate_checkpoint(weights, config)
        self.weights = {}
        shared = weights["model.shared.weight"]
        self.embedding_weight = self.upload(shared, tiled=False)
        self.lm_weight = self.upload(shared, dtype=self.matrix_dtype)
        for name, value in weights.items():
            if "embed_tokens" in name or name in ("model.shared.weight", "lm_head.weight"):
                continue
            if "embed_positions" in name:
                continue
            if value.ndim == 1:
                value = value.reshape(1, 1, 1, -1)
            # Select before upload: never recover BF16 from a BFP8_B tensor.
            dtype = self.matrix_dtype if value.ndim == 2 and name not in self.bf16_matrix_exceptions else ttnn.bfloat16
            self.weights[name] = self.upload(value, dtype=dtype)
        del weights
        print("backend: weights uploaded", flush=True)

    def upload(self, value, *, tiled=True, dtype=None):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype or ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def linear(self, x, name):
        return ttnn.linear(
            x,
            self.weights[name + ".weight"],
            transpose_b=True,
            bias=self.weights.get(name + ".bias"),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.kernel,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def norm(self, x, name):
        return ttnn.layer_norm(
            x,
            epsilon=1e-5,
            weight=self.weights[name + ".weight"],
            bias=self.weights[name + ".bias"],
            compute_kernel_config=self.kernel,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def embedding_host_inputs(self, ids):
        """Prepare token indices and analytic positions outside device capture."""
        # Host work here is integer indexing and nonlearned sinusoidal construction.
        valid = (ids != self.pad).astype(np.int64)
        positions = np.cumsum(valid, axis=-1) * valid + self.pad
        half = self.dim // 2
        frequencies = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / (half - 1)))
        angles = torch.from_numpy(positions).float().unsqueeze(-1) * frequencies
        pos = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if self.dim % 2:
            pos = torch.cat((pos, torch.zeros(*pos.shape[:-1], 1)), dim=-1)
        pos[torch.from_numpy(ids == self.pad)] = 0
        return torch.from_numpy(ids.astype(np.int32)), pos.reshape(1, 1, ids.shape[1], self.dim)

    def embedding_inputs(self, ids):
        indices, positions = self.embedding_host_inputs(ids)
        return self.upload(indices, tiled=False, dtype=ttnn.uint32), self.upload(positions)

    def embed_device(self, indices, positions):
        """Learned embedding and position addition using device inputs only."""
        x = ttnn.embedding(indices, self.embedding_weight, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, 1, indices.shape[-1], self.dim))
        return ttnn.add(ttnn.multiply(x, self.scale), positions)

    def embed(self, ids):
        return self.embed_device(*self.embedding_inputs(ids))

    def mask(self, valid, query_length, *, causal=False):
        allowed = np.broadcast_to(valid.astype(bool)[None, None, :], (1, query_length, valid.size)).copy()
        if causal:
            allowed &= np.arange(valid.size)[None, None, :] <= np.arange(query_length)[None, :, None]
        # Each real decoder query always has at least one valid key.
        values = np.where(allowed, 0.0, -1e9).astype(np.float32)
        return self.upload(torch.from_numpy(values).reshape(1, 1, query_length, valid.size))

    def attention(self, x, memory, prefix, heads, mask, *, cross_kv=None):
        length, source = int(x.shape[-2]), int(memory.shape[-2])
        width = self.dim // heads

        def split(z, n):
            return ttnn.permute(ttnn.reshape(z, (1, n, heads, width)), (0, 2, 1, 3))

        q = split(self.linear(x, prefix + ".q_proj"), length)
        # Only generation supplies this row-local dictionary; self-attention and
        # forward always compute their own K/V. No tensor survives the request.
        if cross_kv is not None and prefix in cross_kv:
            k, v = cross_kv[prefix]
        else:
            k = split(self.linear(memory, prefix + ".k_proj"), source)
            v = split(self.linear(memory, prefix + ".v_proj"), source)
            if cross_kv is not None:
                cross_kv[prefix] = (k, v)
        q = ttnn.multiply(q, width**-0.5)
        if prefix.startswith("model.decoder."):
            result = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=False,
                scale=1.0,
                program_config=self.decoder_sdpa_config,
                compute_kernel_config=self.kernel,
            )
        else:
            scores = ttnn.matmul(q, k, transpose_b=True, compute_kernel_config=self.kernel)
            scores = ttnn.add(scores, mask)
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.kernel, numeric_stable=True)
            result = ttnn.matmul(probs, v, compute_kernel_config=self.kernel)
        result = ttnn.reshape(ttnn.permute(result, (0, 2, 1, 3)), (1, 1, length, self.dim))
        return self.linear(result, prefix + ".out_proj")

    @staticmethod
    def padded(ids, pad):
        length = ids.shape[1]
        padded_length = (length + 31) // 32 * 32
        return np.pad(ids, ((0, 0), (0, padded_length - length)), constant_values=pad)

    def encode(self, ids, attention_mask):
        ids = self.padded(ids, self.pad)
        valid = np.pad(attention_mask[0], (0, ids.shape[1] - attention_mask.shape[1]))
        mask = self.mask(valid, ids.shape[1])
        x = self.embed(ids)
        for layer in range(int(self.config["encoder_layers"])):
            p = "model.encoder.layers." + str(layer)
            normed = self.norm(x, p + ".self_attn_layer_norm")
            x = ttnn.add(
                x, self.attention(normed, normed, p + ".self_attn", int(self.config["encoder_attention_heads"]), mask)
            )
            normed = self.norm(x, p + ".final_layer_norm")
            ff = self.linear(ttnn.relu(self.linear(normed, p + ".fc1")), p + ".fc2")
            x = ttnn.add(x, ff)
        return self.norm(x, "model.encoder.layer_norm"), valid

    def decoder_memory(self, encoder, valid):
        # Extra fully masked tiles change decoder SDPA rounding. Keep every
        # position through the last unmasked key, including all interior masks.
        # Public validation rejects fully masked rows; leave internal calls intact.
        positions = np.flatnonzero(valid)
        if positions.size:
            length = (int(positions[-1]) + 32) // 32 * 32
            if length < len(valid):
                shape = tuple(encoder.shape)
                encoder = ttnn.slice(encoder, (0,) * len(shape), shape[:-2] + (length, shape[-1]))
                valid = valid[:length]
        return encoder, valid

    def decoder_body(self, indices, positions, encoder, causal, cross, *, cross_kv=None):
        """Device-only full-prefix graph, ending at the decoder final norm.

        Inputs already have the original padded prefix and trimmed source extents.
        Capture callers must prepare inputs and populate request-owned cross-KV
        before capture; this method neither uploads inputs nor reads outputs.
        Dynamic row selection and vocabulary projection belong to decode().
        """
        x = self.embed_device(indices, positions)
        heads = int(self.config["decoder_attention_heads"])
        for layer in range(int(self.config["decoder_layers"])):
            p = "model.decoder.layers." + str(layer)
            normed = self.norm(x, p + ".self_attn_layer_norm")
            x = ttnn.add(x, self.attention(normed, normed, p + ".self_attn", heads, causal))
            normed = self.norm(x, p + ".encoder_attn_layer_norm")
            x = ttnn.add(x, self.attention(normed, encoder, p + ".encoder_attn", heads, cross, cross_kv=cross_kv))
            normed = self.norm(x, p + ".final_layer_norm")
            x = ttnn.add(x, self.linear(ttnn.relu(self.linear(normed, p + ".fc1")), p + ".fc2"))
        return self.norm(x, "model.decoder.layer_norm")

    def last_warmup_reuse(self):
        """Opt in for one uninterrupted enabled device program-cache lifetime.

        The caller must invalidate before clearing/disabling the cache or
        changing configuration, and exit before replacing/closing the device.
        No cache-count heuristic can establish this lifetime.
        """
        from models.experimental.nllb.tt.trace_decode import LastWarmupReuse

        return LastWarmupReuse(self)

    def invalidate_last_warmup(self):
        """Forget preparation metadata BEFORE owner-controlled cache mutation."""
        self._guard_trace_ownership()
        if getattr(self, "_decode_trace", None) is not None or getattr(self, "_last_warmup_owners", ()):
            raise RuntimeError("Cannot invalidate LAST warmup during active trace ownership")
        scope = getattr(self, "_last_warmup_reuse", None)
        if scope is not None:
            scope.variants.clear()

    def _guard_trace_ownership(self):
        if getattr(self, "_trace_failures", ()):
            raise RuntimeError("Prior native trace cleanup unresolved; caller device retained")

    def decode(self, ids, encoder, valid, *, final_token_only=False, cross_kv=None):
        self._guard_trace_ownership()
        owner = getattr(self, "_decode_trace", None)
        if owner is not None and owner.cross_kv is cross_kv and final_token_only:
            return owner.decode(ids)
        encoder, valid = self.decoder_memory(encoder, valid)
        real_length = ids.shape[1]
        ids = self.padded(ids, self.pad)
        length = ids.shape[1]
        decoder_valid = np.arange(length) < real_length
        causal = self.mask(decoder_valid, length, causal=True)
        cross = self.mask(valid, length)
        indices, positions = self.embedding_inputs(ids)
        x = self.decoder_body(indices, positions, encoder, causal, cross, cross_kv=cross_kv)
        return self.project_decoder(x, real_length, final_token_only=final_token_only)

    def project_decoder(self, x, real_length, *, final_token_only=False):
        length = int(x.shape[-2])
        last_position = real_length - 1
        if final_token_only and self.generation_projection == "last":
            # Generation processes each batch row separately. Select its last
            # actual prefix position, independent of PAD token values/mask sums.
            # Keep the learned projection on TT and explicitly zero tile padding.
            x = ttnn.slice(x, (0, 0, last_position, 0), (1, 1, real_length, self.dim), pad_value=0.0)
            last_position = 0
        logits = ttnn.linear(
            x, self.lm_weight, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=self.kernel
        )
        if final_token_only:
            # Select the last real query, never a padded row. Row-major layout
            # removes tile-height padding before device-to-host readback.
            logits = ttnn.slice(logits, (0, 0, last_position, 0), (1, 1, last_position + 1, self.vocab))
            logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            return ttnn.to_torch(logits).float().numpy().reshape(1, 1, self.vocab).copy()
        return ttnn.to_torch(logits).float().numpy().reshape(1, length, self.vocab)[:, :real_length].copy()

    def project_selected_device(self, x):
        """Unchanged LAST projection and readback layout, entirely on device."""
        assert tuple(x.shape) == (1, 1, 1, self.dim)
        assert tuple(x.padded_shape) == (1, 1, 32, self.dim)
        logits = ttnn.linear(
            x, self.lm_weight, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=self.kernel
        )
        logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, 1, self.vocab))
        return ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)

    def read_decoder_row(self, logits):
        return ttnn.to_torch(logits).float().numpy().reshape(1, 1, self.vocab).copy()

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        self._guard_trace_ownership()
        validate_inputs(input_ids, attention_mask, self.config)
        token_array(
            decoder_input_ids,
            "decoder_input_ids",
            self.vocab,
            min(64, self.config["max_position_embeddings"]),
            input_ids.shape[0],
        )
        encoders, logits = [], []
        for row in range(input_ids.shape[0]):
            encoder, valid = self.encode(input_ids[row : row + 1], attention_mask[row : row + 1])
            encoders.append(
                ttnn.to_torch(encoder).float().numpy().reshape(1, -1, self.dim)[:, : input_ids.shape[1]].copy()
            )
            logits.append(self.decode(decoder_input_ids[row : row + 1], encoder, valid))
        return {"encoder": np.concatenate(encoders), "logits": np.concatenate(logits)}

    def generate(self, input_ids, attention_mask, target_id, max_new_tokens):
        self._guard_trace_ownership()
        validate_inputs(input_ids, attention_mask, self.config)
        target_id = integer(target_id, "target_id", 0, self.vocab - 1)
        # Config identifies reserved tokens, but supplies no language-ID mapping.
        # Reject declared PAD/EOS/start/BOS/UNK; other in-range IDs remain the
        # caller's responsibility. Do not guess a contiguous language range.
        for name in ("pad_token_id", "eos_token_id", "decoder_start_token_id", "bos_token_id", "unk_token_id"):
            if target_id == self.config.get(name):
                raise ValueError(f"target_id {target_id} is reserved ({name}), not a language token")
        max_new_tokens = integer(
            max_new_tokens, "max_new_tokens", 1, min(256, self.config["max_position_embeddings"] - 1)
        )
        decode = self.decode
        if (
            getattr(decode, "__func__", None) is _CANONICAL_DECODE
            and getattr(decode, "__self__", None) is self
            and input_ids.shape[0] > 1
            and max_new_tokens > 1
            and getattr(self, "trace_decoder_enabled", False)
            and getattr(self, "generation_projection", "full") == "last"
            and self.precision_policy["mode"] in ("bf16", "bfp8_b")
            and all(self.config[key] == value for key, value in _DISTILLED_600M.items())
        ):
            from models.experimental.nllb.tt.trace_decode import generate_packed

            return generate_packed(self, input_ids, attention_mask, target_id, max_new_tokens)
        outputs = []
        for row in range(input_ids.shape[0]):
            encoder, valid = self.encode(input_ids[row : row + 1], attention_mask[row : row + 1])
            tokens = [2, int(target_id)]
            cross_kv = {}
            owner = None
            if getattr(self, "trace_decoder_enabled", False) and max_new_tokens > 1:
                from models.experimental.nllb.tt.trace_decode import DecoderTrace, ProjectedDecoderTrace

                trace_type = (
                    ProjectedDecoderTrace if getattr(self, "generation_projection", "full") == "last" else DecoderTrace
                )
                owner = trace_type(self, encoder, valid, cross_kv)
                self._decode_trace = owner
            try:
                for _ in range(1, max_new_tokens):
                    logits = self.decode(
                        np.array([tokens], dtype=np.int64), encoder, valid, final_token_only=True, cross_kv=cross_kv
                    )
                    last_logits = logits[0, -1]
                    if not np.isfinite(last_logits).all():
                        raise FloatingPointError("Nonfinite last-token logits: NaN or infinity before greedy argmax")
                    token = int(np.argmax(last_logits))
                    tokens.append(token)
                    if token == 2:
                        break
            finally:
                # Also release references on decoder failure, before another row.
                if owner is not None:
                    import sys

                    try:
                        owner.close(preserve_exception=sys.exc_info()[0] is not None)
                    finally:
                        self._decode_trace = None
                if owner is None or not owner.unresolved:
                    cross_kv.clear()
            outputs.append(tokens)
        result = np.full((len(outputs), max(map(len, outputs))), self.pad, dtype=np.int64)
        for row, tokens in enumerate(outputs):
            result[row, : len(tokens)] = tokens
        return result


# Capture once: later class replacement must not approve a custom decode.
_CANONICAL_DECODE = Backend.decode
