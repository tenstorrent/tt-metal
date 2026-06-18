# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full Qwen3.5 text model for Blackhole P150 — a ttnn port of transformers' Qwen3_5TextModel.

Assembly mirrors the HF golden exactly:

    input_ids -> embed_tokens -> N x Qwen35DecoderLayer -> final RMSNorm -> last_hidden_state

plus the LM head (HF keeps that in Qwen3_5ForCausalLM) so the model produces logits and can run
tokens->tokens end to end via generate(). Each decoder layer is either a full (softmax) attention
or a Gated DeltaNet block, chosen per index by args.is_full_attention_layer — the per-layer wiring
lives in tt/layer.py and the token mixers / MLP / norms each have their own unit tests; this file is
only the embedding + layer-stack + norm + head glue and the prefill/decode/generate drivers.

Scope (deliberately B=1 single-stream): prefill fills one user's KV/GDN state, decode steps that one
stream. This is the validated single-device path and the only one where the GDN state buffers (sized
to max_batch_size, overwritten wholesale by forward_prefill) line up. Batched multi-user serving and
vLLM/paged integration are intentionally NOT here — see the old git history for the deprecated
TP-fork / masked-bucket / chunked-trace / paged machinery this file replaces.

See Qwen3_5TextModel / Qwen3_5ForCausalLM in transformers.models.qwen3_5.modeling_qwen3_5.
"""
import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.layer import Qwen35DecoderLayer
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding


class Qwen35Model(LightweightModule):
    """Qwen3.5 text-only language model (embedding + hybrid decoder stack + norm + LM head).

    Usage:
        model = Qwen35Model.from_pretrained(mesh_device)        # HF_MODEL env var = the checkpoint
        out_ids = model.generate(prompt_ids, max_new_tokens=32)
    """

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        super().__init__()
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device  # alias some callers (demo/common) read
        self.configuration = args  # alias some callers read for max_seq_len etc.
        self.num_devices = mesh_device.get_num_devices()
        self.vocab_size = args.vocab_size

        # CCL collective for the multi-device reduce-scatters/all-gathers the layers + final norm
        # ride; None on a single device, where every CCL op short-circuits to a no-op.
        if self.num_devices > 1:
            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None

        # Embedding — framework Embedding, which shards the hidden dim (ShardTensor2dMesh dims=(None,3)).
        # On a single device that replicates the full table; on a mesh it fractures the residual stream
        # right at the source, which is exactly the layout the column-parallel decoder layers consume.
        self.embd = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=tensor_cache_path,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        # Decoder stack — one Qwen35DecoderLayer per index; each picks its own token-mixer kind.
        logger.info(f"Loading {args.n_layers} transformer layers...")
        self.layers = [
            Qwen35DecoderLayer(mesh_device, args, state_dict, i, tensor_cache_path, tt_ccl=self.tt_ccl)
            for i in tqdm(range(args.n_layers), desc="Loading layers")
        ]

        # Final norm — framework RMSNorm (add_unit_offset bakes Qwen3_5RMSNorm's (1 + weight) scale).
        # On TP the post-last-layer residual is fractured along the hidden dim, so wrap in
        # DistributedNorm to reduce stats + all-gather back to the full dim the LM head needs.
        self.norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key="norm",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=self.tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            self.norm = DistributedNorm(self.norm, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy)

        # LM head — a plain [dim, vocab] matmul weight. REPLICATED on a mesh (full vocab on every
        # device) so the gathered full-dim norm output yields full logits without a vocab gather; a
        # vocab-sharded head is a later memory optimization. bf8 weight matches the validated 9B path.
        lm_head_weight = state_dict["output.weight"].T.contiguous()  # [vocab, dim] -> [dim, vocab]
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / "output.weight") if tensor_cache_path else None,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if self.num_devices > 1 else {}),
        )

    @classmethod
    def from_pretrained(cls, device, max_batch_size=1, max_seq_len=2048, n_layers=None, hf_model=None):
        """Build the model from the HF_MODEL checkpoint (env var = the single source of truth).

        hf_model, if given, sets HF_MODEL first; n_layers, if given, truncates the stack (used by
        tests to keep host RAM low). Mirrors tt/common.create_tt_model's load+remap+cache sequence.
        """
        import os

        if hf_model is not None:
            os.environ["HF_MODEL"] = hf_model

        args = Qwen35ModelArgs(mesh_device=device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
        if n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]

        logger.info("Loading + remapping weights via Qwen35ModelArgs.load_state_dict()...")
        state_dict = args.load_state_dict()
        return cls(device, args, state_dict, tensor_cache_path=args.weight_cache_path())

    # ── State ────────────────────────────────────────────────────────────────────────
    def reset_state(self):
        """Zero the per-layer recurrent state before a new sequence.

        Only the GDN layers carry state that PERSISTS across sequences (its conv + recurrent
        buffers accumulate), so they must be reset; the attention KV cache needs no reset because
        prefill overwrites it wholesale via fill_cache. Called at the top of generate().
        """
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.reset_recurrent_state()
                layer.attention.reset_conv_state()

    # ── Core forward (Qwen3_5TextModel parity) ───────────────────────────────────────
    def forward(self, tokens, mode="decode", positions=None, user_id=0):
        """ttnn analog of Qwen3_5TextModel.forward: embed -> rope -> layers -> final norm.

        Returns the normed last_hidden_state in the layers' residual layout — [1, 1, S, dim] for
        prefill, [1, 1, B, dim] for decode (hidden dim fractured on TP, gathered to full by the
        final DistributedNorm). tokens is a host int tensor: [1, S] for prefill, [B, 1] for decode.
        RoPE cos/sin and the decode position tensor are built here from rope_tp and threaded into
        every layer (the GDN layers ignore them; only full-attention consumes them).
        """
        if mode == "prefill":
            assert tokens.shape[0] == 1, "prefill is single-stream (B=1); see module docstring"
            x = self._embed_token_row(tokens)  # [1, 1, S, dim]
            seq_len = tokens.shape[-1]
            cos, sin = rot_mats_prefill(self.device, self.args.rope_head_dim, seq_len, self.args.rope_theta)
            position_tensor = None
        else:
            batch = tokens.shape[0]
            # Embed the B decode tokens as one [1, B] row so the reshape to the [1,1,B,dim] decode
            # residual layout is metadata-only (last two dims stay (B, dim)).
            x = self._embed_token_row(tokens.reshape(1, batch))
            cos, sin = rot_mats_decode(
                self.device, self.args.rope_head_dim, self.args.max_seq_len, self.args.rope_theta, positions
            )
            # Per-user decode positions drive the KV-cache update index + decode RoPE; replicated so
            # every device's local head shard updates at the same slot.
            position_tensor = ttnn.from_torch(
                positions.to(torch.int32),
                dtype=ttnn.int32,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )

        for layer in self.layers:
            # Dispatch to the phase-specific entry point; the per-layer forward split mirrors the token
            # mixers' own forward_prefill/forward_decode and keeps each call to just the args that phase
            # uses (prefill carries the per-user KV-fill id, decode the per-user KV-cache position).
            if mode == "prefill":
                x = layer.forward_prefill(x, cos=cos, sin=sin, user_id=user_id)
            else:
                x = layer.forward_decode(x, position_tensor=position_tensor, cos=cos, sin=sin)

        norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        return self.norm(x, mode=norm_mode)

    def lm_head(self, hidden):
        """Project the (full-dim) normed hidden state to vocab logits via the replicated head."""
        return ttnn.linear(hidden, self.lm_head_weight)

    # ── Host<->device + token-row helpers ────────────────────────────────────────────
    def _embed_token_row(self, token_row):
        """Embed a [1, N] host token row into the [1, 1, N, dim] residual layout.

        Mirrors the framework prefill path exactly: tokens go in as a [1,1,1,N] ROW_MAJOR uint32
        tensor (the layout ttnn.embedding indexes) replicated to the mesh, and the embedded result is
        unsqueezed back to 4D — [1, 1, N, dim] with the hidden dim fractured on TP. Feeding the wrong
        input shape/layout here silently indexes garbage rows, so this must match the framework.
        """
        tok = ttnn.from_torch(
            token_row.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return ttnn.unsqueeze_to_4D(self.embd(tok))  # [1, 1, N, dim]

    def _logits_to_torch(self, logits, n_rows):
        """Bring replicated device logits [1, 1, n_rows, vocab] back to host torch [n_rows, vocab].

        The head is replicated, so every device holds identical logits — take one replica (dim-0
        concat then [0]) on a mesh, or read directly on a single device. The [:vocab_size] guards
        against any tile padding on the vocab dim.
        """
        if self.num_devices > 1:
            t = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))[0]
        else:
            t = ttnn.to_torch(logits)
        return t.reshape(n_rows, -1)[:, : self.vocab_size].float()

    def argmax_on_device(self, logits):
        """Greedy token select done ON DEVICE, so decode reads back one id instead of the full
        248K-wide logit row (the ~15 ms/token readback at TP=4). Used by decode() and by the captured
        decode graph in demo/trace_runner.py. The LM head is replicated, so every device already holds
        the whole vocab — no gather is needed (unlike tt_transformers' sharded head, which all-gathers
        first). untilize -> ROW_MAJOR enables argmax's multi-core last-dim path; vocab (248320) is
        tile-aligned, so there are no padding columns that could win the argmax. Returns a replicated
        [1, 1, n_rows, 1] uint32 id; read one replica back with ttnn.get_device_tensors(tok)[0].
        """
        logits_rm = ttnn.untilize(logits, use_multicore=True)
        return ttnn.argmax(logits_rm, dim=3, keepdim=True, use_multicore=True)

    # ── Public prefill / decode / generate ───────────────────────────────────────────
    def prefill(self, tokens, user_id=0, valid_len=None):
        """Prefill one stream's prompt and return host logits [vocab] at the last real position.

        tokens: host int tensor [1, S]. Runs the full sequence through every layer (filling the
        attention KV cache + GDN conv/recurrent state for this stream), then norms+heads only the
        last position — the only logit greedy generation needs.

        valid_len lets the caller right-pad tokens to a tile multiple for the GDN chunk kernel (see
        generate) while still reading the causal logit at the LAST REAL token (valid_len-1): the
        attention is causal and the GDN delta-rule is lower-triangular, so trailing pad tokens cannot
        influence position valid_len-1. Defaults to the full length (no padding), the test path.
        """
        hidden = self.forward(tokens, mode="prefill", user_id=user_id)  # [1, 1, S, dim]
        idx = (tokens.shape[-1] if valid_len is None else valid_len) - 1
        last = hidden[:, :, idx : idx + 1, :]  # [1, 1, 1, dim] — last REAL position
        logits = self.lm_head(last)  # [1, 1, 1, vocab]
        return self._logits_to_torch(logits, n_rows=1).reshape(-1)

    def decode(self, tokens, positions, return_token=True):
        """Decode one step for B users; return the greedy next-token id(s) OR the raw logits.

        return_token (default True) picks what crosses back from one decode forward:
          * True  -> host int32 [B]: the argmax is done ON DEVICE (argmax_on_device) so only the id
            crosses back, not the full 248K-wide logit row — that readback was ~15 ms/token at TP=4.
            This is the generation path (demo / generate()), which only ever needs the token.
          * False -> host float [B, vocab]: the raw logits, for PCC against the HF golden. The whole
            vocab row crosses back here — that cost is exactly what return_token=True avoids, so it's
            opt-in for tests that need the numerics rather than the picked token.

        Either way it is ONE forward, so the per-user KV cache + GDN state advance exactly once and the
        prefill->decode hand-off stays in lock-step. tokens: host int [B, 1]; positions: host int [B]
        (absolute position of each user's token), continuing from the state prefill / prior steps left.
        """
        hidden = self.forward(tokens, mode="decode", positions=positions)  # [1, 1, B, dim]
        logits = self.lm_head(hidden)  # [1, 1, B, vocab], replicated on a mesh
        if not return_token:
            return self._logits_to_torch(logits, n_rows=tokens.shape[0])  # host [B, vocab]
        tok = self.argmax_on_device(logits)  # [1, 1, B, 1] uint32, replicated on device
        # The id is replicated, so read ONE device's copy -> B ints cross the bus, not B*vocab.
        return ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1).to(torch.int32)

    def generate(self, prompt_ids, max_new_tokens=20, eos_token_id=None):
        """Greedy end-to-end generation for a single stream (B=1). Returns the list of new token ids.

        Resets state, prefills the prompt, then argmax-decodes one token at a time, advancing the
        absolute position so the attention KV cache reads grow with the sequence. eos_token_id, if
        given, stops early.
        """
        prompt = torch.as_tensor(prompt_ids, dtype=torch.int32).reshape(1, -1)
        valid_len = prompt.shape[1]

        # The GDN chunk-prefill kernel requires a tile-aligned sequence (seq_len % 32 == 0), so
        # right-pad the prompt with token 0 up to the next multiple of 32. Right-padding is safe for
        # the next-token logit (read at valid_len-1; causal attention + lower-triangular GDN ignore
        # trailing tokens), and we then continue DECODE from the padded length T_pad — not valid_len —
        # so the position the decode KV-cache update / GDN state advance from matches the state prefill
        # actually captured (the new GDN prefill folds its conv/recurrent state over the full padded
        # sequence). The pad (<32 tokens) sits between prompt and generation; for short prompts this is
        # a small approximation, removable by threading valid_len into the GDN to capture state at the
        # real boundary (see tt/gdn/gdn.py forward_prefill — the old model.py did this via valid_len).
        pad = (-valid_len) % 32
        if pad:
            prompt = torch.cat([prompt, torch.zeros(1, pad, dtype=torch.int32)], dim=1)
        seq_len = prompt.shape[1]  # T_pad — the padded prefill length
        assert (
            seq_len + max_new_tokens <= self.args.max_seq_len
        ), f"padded prompt ({seq_len}) + new tokens ({max_new_tokens}) exceeds max_seq_len ({self.args.max_seq_len})"

        self.reset_state()
        logits = self.prefill(prompt, valid_len=valid_len)  # logit at the last REAL token -> first new token
        next_id = int(torch.argmax(logits).item())
        out = [next_id]

        # Each decode feeds the just-produced token at its absolute position (the first new token
        # sits at position seq_len = T_pad) and predicts the next one.
        cur_pos = seq_len
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_id == eos_token_id:
                break
            # decode argmaxes on device and reads back just the id (see argmax_on_device).
            next_id = int(
                self.decode(torch.tensor([[next_id]], dtype=torch.int32), torch.tensor([cur_pos], dtype=torch.int32))[0]
            )
            out.append(next_id)
            cur_pos += 1
        return out
