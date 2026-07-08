# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Higgs Audio v2 generation on TTNN.

The LLM backbone (prefill + autoregressive audio-token decode) runs on our TTNN
``HiggsAudioTTModel``; the HF ``HiggsAudioV2Processor`` is reused for I/O only —
chat templating, reference-audio encoding, and audio-token -> waveform decoding
(the DAC-based ``HiggsAudioV2TokenizerModel`` codec at /data/hf_cache/higgs/tokenizer).

Generation protocol (mirrors HF generate's audio_sequences format so we can hand
the result straight to ``processor.batch_decode``):
  1. Build input_ids from a chat conversation (ends at <|audio_out_bos|>).
  2. TTNN prefill over the prompt (text, + spliced reference-audio embeddings for
     voice cloning) -> first audio-row logits.
  3. Greedy-argmax each step through the delay-pattern state machine
     (tt/audio_decode.py), feeding the selected row back into ``decode_step_audio``
     until the all-EOS row (finished) or max_new_tokens.
  4. Stack rows as [1, T, 8] with a leading all-BOS row -> processor.batch_decode.
"""
from __future__ import annotations

import os
import pathlib
import time

import torch
from loguru import logger

import ttnn
from models.demos.audio.higgs_audio_v2.tt.audio_decode import (
    apply_delay_pattern_to_selected_audio_tokens,
    initialize_delay_pattern_state,
)
from models.demos.audio.higgs_audio_v2.tt.model import HiggsAudioTTModel
from models.demos.audio.higgs_audio_v2.tt.model_args import BASE_TEXT_MODEL, HiggsModelArgs
from models.demos.audio.higgs_audio_v2.tt.precision_presets import build_precision
from models.demos.audio.higgs_audio_v2.tt.reference import HiggsAudioV2Config, load_higgs_v2_state_dict
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup

HIGGS_MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"
SAMPLING_RATE = 24000


def resolve_model_dir(model_dir: str | None = None) -> str:
    """Locate the Higgs assets (config.json, transformers-native weights, tokenizer/).

    Resolution order, so no absolute path is ever hard-coded:
      1. explicit ``model_dir`` arg or ``HIGGS_MODEL_DIR`` env — a pre-staged local
         directory (offline / IRD containers / a lab box's /data cache);
      2. an HF-hub snapshot of ``HIGGS_MODEL_ID``, cached under the standard HF cache
         (``HF_HOME`` / ``HF_HUB_CACHE``; ``HF_HUB_OFFLINE=1`` resolves cache-only).

    Returns a local directory usable by ``AutoProcessor.from_pretrained``,
    ``HiggsAudioV2Config.from_json`` and ``load_higgs_v2_state_dict``.
    """
    model_dir = model_dir or os.environ.get("HIGGS_MODEL_DIR")
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download

    return snapshot_download(HIGGS_MODEL_ID)


class _DelayCfg:
    def __init__(self, args):
        self.audio_num_codebooks = args.audio_num_codebooks
        self.audio_stream_bos_id = args.audio_stream_bos_id
        self.audio_stream_eos_id = args.audio_stream_eos_id
        self.use_delay_pattern = True


class HiggsAudioTTSGenerator:
    def __init__(self, mesh_device, precision="performance", model_dir=None, max_batch_size=1):
        from transformers import AutoProcessor

        self.mesh_device = mesh_device
        self.model_dir = resolve_model_dir(model_dir)
        self.max_batch_size = max_batch_size
        self.processor = AutoProcessor.from_pretrained(self.model_dir)

        higgs_cfg = HiggsAudioV2Config.from_json(pathlib.Path(self.model_dir) / "config.json")
        opt = build_precision(precision, higgs_cfg.num_hidden_layers, BASE_TEXT_MODEL)
        self.args = HiggsModelArgs(
            mesh_device=mesh_device,
            higgs_config=higgs_cfg,
            max_batch_size=max_batch_size,
            max_seq_len=1024,
            optimizations=opt,
        )
        self.cfg = _DelayCfg(self.args)
        self.K = self.args.audio_num_codebooks
        self.cb_size = self.args.audio_codebook_size

        _, state_dict = load_higgs_v2_state_dict(self.model_dir)
        tt_ccl = TT_CCL(mesh_device)
        RopeCls = HfRotarySetup if self.args.use_hf_rope else RotarySetup
        self.rope_setup = RopeCls(
            device=mesh_device,
            batch_size=self.args.max_batch_size,
            head_dim=self.args.head_dim,
            max_seq_len=self.args.max_seq_len,
            rope_theta=self.args.rope_theta,
            rope_scaling=self.args.rope_scaling,
            use_qk_fused=self.args.use_qk_fused,
            prefetcher=None,
        )
        self.model = HiggsAudioTTModel(
            args=self.args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            transformation_mats=self.rope_setup.get_both_trans_mats(),
            dtype=ttnn.bfloat8_b,
        )
        logger.info(f"HiggsAudioTTSGenerator ready (precision={precision}, dim={self.args.dim})")

    # ---- prompt building -----------------------------------------------------
    def build_inputs(self, conversation):
        """Return (input_ids[S], audio_input_ids or None) for a chat conversation.

        Text-only conversations (TTS / multi-speaker) -> input_ids only.
        Conversations with an assistant 'audio' turn (voice cloning) -> also the
        reference audio_input_ids the processor encoded.
        """
        enc = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"][0].to(torch.int64)
        audio_in = enc.get("audio_input_ids", None)
        audio_mask = enc.get("audio_input_ids_mask", None)
        return input_ids, audio_in, audio_mask

    # ---- decode helpers ------------------------------------------------------
    def _rot_inputs(self, pos):
        cp = torch.tensor([pos], dtype=torch.int32)
        ridx = self.rope_setup.get_rot_idxs(cp, on_host=True)
        ridx = ttnn.to_device(ridx, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cp_tt = ttnn.from_torch(
            cp,
            device=self.mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
        )
        return cp_tt, self.rope_setup.get_rot_mats(ridx)

    @staticmethod
    def _sample_tokens(logits, temperature, top_k, top_p):
        """Per-codebook sampling. logits [K, cb_size] -> selected ids [K].

        temperature<=0 -> greedy argmax. Otherwise temperature + optional top-k
        + nucleus (top-p), sampled independently per codebook. BOS/EOS ids stay
        in the distribution so the stream can terminate naturally.
        """
        if temperature is None or temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits.float() / temperature
        if top_k and top_k > 0:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p and top_p < 1.0:
            s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
            remove = (cum - torch.softmax(s_logits, dim=-1)) > top_p  # always keep top-1
            s_logits = s_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, s_idx, s_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _ras(self, selected, logits, rows, win_len, max_rep):
        """Repetition-Aware Sampling (Higgs RAS, ras_win_len/ras_win_max_num_repeat).

        For each codebook whose sampled token already appears >= max_rep times in
        the last win_len generated frames (excluding BOS/EOS), resample it from the
        raw logits (no temperature) — breaks phrase-level repetition loops.
        """
        if win_len <= 0 or not rows:
            return selected
        window = torch.stack(rows[-win_len:], dim=0)  # [w, K]
        rep = window == selected.unsqueeze(0)
        rep = rep & (window != self.cfg.audio_stream_bos_id) & (window != self.cfg.audio_stream_eos_id)
        repl = rep.sum(dim=0) >= max_rep  # [K]
        if bool(repl.any()):
            resampled = torch.multinomial(torch.softmax(logits.float(), dim=-1), 1).squeeze(-1)
            selected = torch.where(repl, resampled.to(selected.dtype), selected)
        return selected

    # ---- traced decode --------------------------------------------------------
    def _build_decode_trace(self, start_pos):
        """Capture the audio decode step as one replayable ``ttnn.execute_trace``.

        The whole device forward — on-device audio embedding (add codebook
        offsets -> ttnn.embedding -> sum) + 28 DualFFN blocks + norm + audio LM
        head -> per-codebook logits — is chained on device and captured once, so
        each subsequent step is a single trace replay instead of hundreds of
        host-dispatched ops (the ~58 ms/step overhead the untraced loop pays).
        Sampling + RAS + the delay-pattern state machine stay on host between
        replays (they need the logits and are cheap), so the trace stays generic
        across steps while generation quality is unchanged from the eager path.

        Returns a dict of persistent buffers + closures:
          advance_pos(pos)  -> update the KV position + rope idx (host->device)
          feed_tokens(ids)  -> write the previous step's [K] tokens into cur_tokens
          logits_out        -> the trace's output tensor (read after each replay)
        """
        K, cb = self.K, self.cb_size
        dev = self.mesh_device
        skip_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        norm_cfg = self.args.get_norm_config("lm_head", Mode.DECODE, None)
        cluster_shape = self.args.cluster_shape

        cur_tokens = ttnn.from_torch(
            torch.zeros(1, K, dtype=torch.int32), device=dev, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )
        offsets_dev = ttnn.from_torch(
            (torch.arange(K, dtype=torch.int32) * cb).view(1, K), device=dev, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )
        cp_dev = ttnn.from_torch(
            torch.tensor([start_pos], dtype=torch.int32), device=dev, dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=cluster_shape),
        )
        rope_idxs = self.rope_setup.get_rot_idxs(torch.tensor([start_pos], dtype=torch.int32), on_host=False)

        def step_body():
            ids = ttnn.add(cur_tokens, offsets_dev)
            emb = self.model.audio_embedding(ids)
            h = ttnn.sum(emb, dim=1, keepdim=True)
            h = ttnn.reshape(h, (1, 1, 1, h.shape[-1]))
            ttnn.deallocate(emb)
            h = ttnn.to_memory_config(h, skip_mem_cfg)
            rot_mats = self.rope_setup.get_rot_mats(rope_idxs)
            for blk in self.model.layers:
                h = blk(h, cp_dev, rot_mats, mode=Mode.DECODE, is_audio_token=True)
            h = self.model.norm(h, Mode.DECODE, norm_config=norm_cfg)
            logits = self.model.audio_lm_head(h)
            logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, 1, K * cb))
            return ttnn.reshape(logits, (1, 1, K, cb))

        def advance_pos(pos):
            cp_host = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=cluster_shape),
            )
            ttnn.copy_host_to_device_tensor(cp_host, cp_dev)
            ridx_host = self.rope_setup.get_rot_idxs(torch.tensor([pos], dtype=torch.int32), on_host=True)
            ttnn.copy_host_to_device_tensor(ridx_host, rope_idxs)

        def feed_tokens(next_ids):
            nxt_host = ttnn.from_torch(
                next_ids.view(1, K).to(torch.int32), dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
            )
            ttnn.copy_host_to_device_tensor(nxt_host, cur_tokens)

        # compile (untraced) once, then capture the trace
        advance_pos(start_pos)
        _ = step_body()
        ttnn.synchronize_device(dev)
        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        logits_out = step_body()
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)
        return {
            "trace_id": trace_id, "logits_out": logits_out,
            "advance_pos": advance_pos, "feed_tokens": feed_tokens,
        }

    def _read_logits(self, logits_out):
        K, cb = self.K, self.cb_size
        t = ttnn.to_torch(logits_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        return t.reshape(-1)[: K * cb].reshape(K, cb).float()

    def _build_ondevice_sample_trace(self, start_pos, temperature, top_k, max_new_tokens, seed):
        """Capture a FULLY-ON-DEVICE decode+sample step as one ``ttnn.execute_trace``.

        The whole step runs on device: forward (embedding + 28 DualFFN blocks + norm +
        audio head) -> per-codebook logits -> top-k mask (``topk``/``repeat``/``ge``/``where``)
        -> temperature (``mul``) -> gumbel-max draw (``add`` pre-generated noise + ``argmax``)
        -> delay-pattern startup BOS-stuffing (``mul``/``add`` positional masks) -> feedback
        (``copy`` into cur_tokens). Only the 8 chosen tokens are read back per step -- the
        sampling is NOT done on host. Gumbel noise is pre-generated on host once (a [K, cb]
        slab per step) and streamed into ``noise_cur`` each step: device RNG (``ttnn.rand``)
        is baked at capture time inside a trace, so it can't supply fresh per-step entropy.
        """
        K, cb = self.K, self.cb_size
        dev = self.mesh_device
        skip_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        norm_cfg = self.args.get_norm_config("lm_head", Mode.DECODE, None)
        cluster_shape = self.args.cluster_shape
        rep = ttnn.ReplicateTensorToMesh(dev)
        bos = self.cfg.audio_stream_bos_id
        temp = temperature if (temperature and temperature > 0) else 1.0
        tk = min(top_k, cb) if top_k else cb

        def _u32(t):
            return ttnn.from_torch(
                t.to(torch.int32), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            )

        def _u32h(t):
            return ttnn.from_torch(t.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)

        def _bf16(t):
            return ttnn.from_torch(t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)

        cur_tokens = _u32(torch.zeros(1, K))
        offsets_dev = _u32((torch.arange(K) * cb).view(1, K))
        keep_mask = _u32(torch.ones(1, K))
        stuff_mask = _u32(torch.zeros(1, K))
        cp_dev = ttnn.from_torch(
            torch.tensor([start_pos], dtype=torch.int32), device=dev, dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=cluster_shape),
        )
        rope_idxs = self.rope_setup.get_rot_idxs(torch.tensor([start_pos], dtype=torch.int32), on_host=False)
        inv_temp = _bf16(torch.full((1, 1, K, cb), 1.0 / temp))
        neg_const = _bf16(torch.full((1, 1, K, cb), -1e9))
        # Pre-generate gumbel noise on host: one [K, cb] slab per step, streamed in each step.
        if seed is not None:
            torch.manual_seed(seed)
        u = torch.rand(max_new_tokens, K, cb).clamp_(1e-6, 1 - 1e-6)
        gumbel = (-torch.log(-torch.log(u))).to(torch.bfloat16)
        noise_cur = _bf16(gumbel[0].view(1, 1, K, cb))

        def step_body():
            ids = ttnn.add(cur_tokens, offsets_dev)
            emb = self.model.audio_embedding(ids)
            h = ttnn.sum(emb, dim=1, keepdim=True)
            h = ttnn.reshape(h, (1, 1, 1, h.shape[-1]))
            ttnn.deallocate(emb)
            h = ttnn.to_memory_config(h, skip_mem_cfg)
            rot_mats = self.rope_setup.get_rot_mats(rope_idxs)
            for blk in self.model.layers:
                h = blk(h, cp_dev, rot_mats, mode=Mode.DECODE, is_audio_token=True)
            h = self.model.norm(h, Mode.DECODE, norm_config=norm_cfg)
            lg = self.model.audio_lm_head(h)
            lg = ttnn.slice(lg, (0, 0, 0, 0), (1, 1, 1, K * cb))
            lg = ttnn.reshape(lg, (1, 1, K, cb))
            # top-k: keep logits >= the k-th largest (materialize the threshold to full width
            # to avoid an unreliable broadcast compare), mask the rest to -1e9.
            kth = ttnn.slice(ttnn.topk(lg, tk, dim=-1)[0], (0, 0, 0, tk - 1), (1, 1, K, tk))
            kth = ttnn.repeat(kth, ttnn.Shape([1, 1, 1, cb]))
            lg = ttnn.where(ttnn.ge(lg, kth), lg, neg_const)
            # temperature, then gumbel-max draw = argmax(logits/T + gumbel)
            lg = ttnn.mul(lg, inv_temp)
            lg = ttnn.add(lg, noise_cur)
            nx = ttnn.argmax(lg, dim=-1, keepdim=False)
            nx = ttnn.reshape(nx, (1, K))
            nx = ttnn.to_layout(nx, ttnn.ROW_MAJOR_LAYOUT)
            nx = ttnn.typecast(nx, ttnn.uint32)
            # delay-pattern startup BOS-stuffing: nx*keep + stuff (steady-state = identity)
            nx = ttnn.add(ttnn.mul(nx, keep_mask), stuff_mask)
            ttnn.copy(nx, cur_tokens)
            return nx

        def advance_pos(pos):
            cp_host = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=cluster_shape),
            )
            ttnn.copy_host_to_device_tensor(cp_host, cp_dev, cq_id=0)
            ridx_host = self.rope_setup.get_rot_idxs(torch.tensor([pos], dtype=torch.int32), on_host=True)
            ttnn.copy_host_to_device_tensor(ridx_host, rope_idxs, cq_id=0)

        def set_step(i):
            # positional startup BOS-stuffing masks (codebook c not started until step c),
            # steady-state = identity; plus this step's pre-generated gumbel slab.
            if i < K - 1:
                kv = [1 if c <= i else 0 for c in range(K)]
                sv = [0 if c <= i else bos for c in range(K)]
            else:
                kv = [1] * K
                sv = [0] * K
            ttnn.copy_host_to_device_tensor(_u32h(torch.tensor(kv).view(1, K)), keep_mask, cq_id=0)
            ttnn.copy_host_to_device_tensor(_u32h(torch.tensor(sv).view(1, K)), stuff_mask, cq_id=0)
            gi = min(i, max_new_tokens - 1)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(gumbel[gi].view(1, 1, K, cb), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep),
                noise_cur, cq_id=0,
            )

        def feed_tokens(next_ids):
            ttnn.copy_host_to_device_tensor(_u32h(next_ids.view(1, K)), cur_tokens, cq_id=0)

        advance_pos(start_pos)
        set_step(0)
        _ = step_body()
        ttnn.synchronize_device(dev)
        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        tok_out = step_body()
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)
        return {
            "trace_id": trace_id, "cur_tokens": cur_tokens, "tok_out": tok_out,
            "advance_pos": advance_pos, "set_step": set_step, "feed_tokens": feed_tokens,
        }

    def _decode_ondevice(self, tr, start_pos, max_new_tokens):
        """Run the fully-on-device sample loop; return delay-pattern-processed [K] rows.

        The delay-pattern state machine runs per step so the loop breaks at the TRUE EOS
        (and applies the EOS shutdown ramp) rather than always running max_new_tokens — the
        latter both wasted ~2x compute on early-terminating streams and made the reported
        tok/s divide the truncated row count by the full max_new_tokens decode time.

        Reads back only the 8 chosen tokens/step. By DEFAULT the dispatch is non-blocking
        (``execute_trace(blocking=False)`` with no per-step ``synchronize``): the ``to_torch``
        read of ``cur_tokens`` is queued after the trace on the SAME command queue, so it is
        correctly ordered against the trace's in-place write — no race, no barrier needed. This
        is ~4% faster than blocking and produces bit-identical tokens (verified across seeds,
        incl. early-EOS). Set ``HIGGS_ONDEV_BLOCKING=1`` to force the blocking path.

        What does NOT work is OVERLAPPING the read with the next step: a non-blocking snapshot
        into a separate per-slot buffer races the trace's ``cur`` write and corrupts frames; a
        2nd command queue orders it correctly but measured slower (event overhead > the <1 ms
        read it overlaps). The default here does not overlap — it just skips the redundant
        per-step barrier — so it stays correct.
        """
        dev = self.mesh_device
        K = self.K
        # Default non-blocking (see docstring); HIGGS_ONDEV_BLOCKING=1 forces blocking.
        blocking = os.environ.get("HIGGS_ONDEV_BLOCKING", "0") == "1"
        bos_row = torch.full((K,), self.cfg.audio_stream_bos_id, dtype=torch.long)
        num_delay, num_rem = initialize_delay_pattern_state(bos_row.view(K, 1), self.cfg)
        tr["feed_tokens"](bos_row)
        rows = []
        finished = False
        for i in range(max_new_tokens):
            tr["set_step"](i)
            tr["advance_pos"](start_pos + i)
            ttnn.execute_trace(dev, tr["trace_id"], cq_id=0, blocking=blocking)
            if blocking:
                ttnn.synchronize_device(dev)
            r = ttnn.to_torch(tr["cur_tokens"], mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0)).reshape(-1)[:K].long()
            # Authoritative EOS via the delay-pattern state machine (also applies the EOS
            # shutdown ramp). The raw all-EOS heuristic is unreliable under sampling — EOS
            # staggers across codebooks and never cleanly fires — so without this the loop
            # runs the full max_new_tokens and wastes ~2x compute on early-terminating streams.
            nxt, _active, num_delay, num_rem, finished = apply_delay_pattern_to_selected_audio_tokens(
                r, self.cfg, num_delay, num_rem
            )
            rows.append(nxt.to(torch.long))
            if finished:
                break
        return rows, finished

    @torch.no_grad()
    def generate(
        self,
        conversation,
        max_new_tokens=750,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        seed=1234,
        silence_patience=32,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        use_trace=True,
        ondevice_sample=True,
    ):
        """Run prefill + free-running greedy audio decode.

        Returns audio_input_ids as a torch.LongTensor [1, T, K] in the delay-
        pattern format (leading all-BOS row, generated rows, trailing all-EOS
        row) ready for ``processor.batch_decode``.
        """
        input_ids, audio_in, audio_mask = self.build_inputs(conversation)
        if audio_in is not None:
            logger.info(f"reference audio present: {tuple(audio_in.shape)} (voice cloning)")
        S = input_ids.shape[0]

        # Prefill populates the KV cache; its audio-head output at the text
        # <|audio_out_bos|> position is NOT a content prediction (it's degenerate),
        # so we ignore it. Generation starts by feeding the all-BOS audio frame
        # (the canonical first audio row, == fixture row 0) at pos=S — exactly the
        # protocol the teacher-forced accuracy gate validated at 0.967.
        if seed is not None:
            torch.manual_seed(seed)
        greedy = (temperature is None) or (temperature <= 0)
        logger.info(
            f"decode: {'greedy' if greedy else f'sampling T={temperature} top_k={top_k} top_p={top_p}'} "
            f"RAS(win={ras_win_len}, max_rep={ras_win_max_num_repeat})"
        )
        _t_pre = time.perf_counter()
        _ = self.model.prefill_text(
            input_ids,
            self.rope_setup,
            audio_input_ids=audio_in,
            audio_input_ids_mask=audio_mask,
            audio_token_id=self.processor.audio_token_id,
        )
        prefill_s = time.perf_counter() - _t_pre
        if os.environ.get("HIGGS_DEBUG"):
            logger.info(f"[dbg] S={S} input_ids tail={input_ids[-6:].tolist()}")

        bos_row = torch.full((self.K,), self.cfg.audio_stream_bos_id, dtype=torch.long)
        num_delay, num_rem = initialize_delay_pattern_state(bos_row.view(self.K, 1), self.cfg)

        # Fully-on-device path: the forward AND the sampling (temperature/top-k/gumbel-max
        # draw), the delay-pattern startup stuffing, and the feedback all run on device; only
        # the 8 chosen tokens are read back per step (no host logits readback / host sampling).
        if ondevice_sample:
            _t_tr = time.perf_counter()
            tr = self._build_ondevice_sample_trace(S, temperature, top_k, max_new_tokens, seed)
            trace_build_s = time.perf_counter() - _t_tr
            logger.info("decode: FULLY on-device sample (forward+top-k+temp+gumbel-max+delay+feedback on device)")
            _t_dec = time.perf_counter()
            try:
                # _decode_ondevice now applies the delay-pattern state machine in-loop and
                # breaks at the true EOS, so its returned rows are final and decode_s covers
                # only the steps actually run (correct tok/s for early-terminating streams).
                rows, finished = self._decode_ondevice(tr, S, max_new_tokens)
            finally:
                ttnn.release_trace(self.mesh_device, tr["trace_id"])
            decode_s = time.perf_counter() - _t_dec
            self._last_timing = {
                "prefill_s": prefill_s, "trace_build_s": trace_build_s, "decode_s": decode_s,
                "rows": len(rows), "decode_tok_per_s": (len(rows) / decode_s) if decode_s > 0 else 0.0,
            }
            logger.info(
                f"timing: prefill {prefill_s*1e3:.0f}ms  trace-build {trace_build_s*1e3:.0f}ms  "
                f"decode {decode_s:.2f}s ({self._last_timing['decode_tok_per_s']:.1f} tok/s over {len(rows)} rows, "
                f"blocking on-device readback)"
            )
            if not finished:
                logger.warning("no natural EOS (max_new_tokens); appending EOS row")
                rows.append(torch.full((self.K,), self.cfg.audio_stream_eos_id, dtype=torch.long))
            audio_seq = torch.stack([bos_row] + rows, dim=0).unsqueeze(0)
            logger.info(f"generated {len(rows)} audio rows (finished={finished}) -> audio_seq {tuple(audio_seq.shape)}")
            return audio_seq

        rows = []  # generated delay-pattern rows ([K] each), rows[0] follows the all-BOS frame
        cur_in = bos_row
        pos = S
        finished = False
        prev_row = None
        repeat = 0  # consecutive identical rows -> silence/degeneration

        # Traced path: the device decode step is captured once and replayed, so the
        # demo runs on the traced hot loop (the untraced eager path is kept as a
        # fallback / for the accuracy harness). Sampling stays on host between replays.
        _t_tr = time.perf_counter()
        tr = self._build_decode_trace(S) if use_trace else None
        if use_trace:
            tr["feed_tokens"](bos_row)  # seed with the all-BOS frame at pos=S
            logger.info("decode: traced device step + host sampling")
        trace_build_s = time.perf_counter() - _t_tr if use_trace else 0.0

        _t_dec = time.perf_counter()
        try:
            for step in range(max_new_tokens):
                if use_trace:
                    tr["advance_pos"](pos)
                    ttnn.execute_trace(self.mesh_device, tr["trace_id"], cq_id=0, blocking=False)
                    ttnn.synchronize_device(self.mesh_device)
                    logits = self._read_logits(tr["logits_out"])
                else:
                    cp_tt, rot_mats = self._rot_inputs(pos)
                    logits = self.model.decode_step_audio(cur_in.clamp(min=0, max=self.cb_size - 1), cp_tt, rot_mats)
                selected = self._sample_tokens(logits, temperature, top_k, top_p)
                selected = self._ras(selected, logits, rows, ras_win_len, ras_win_max_num_repeat)
                nxt, _active, num_delay, num_rem, finished = apply_delay_pattern_to_selected_audio_tokens(
                    selected, self.cfg, num_delay, num_rem
                )
                nxt = nxt.to(torch.long)
                rows.append(nxt)
                if os.environ.get("HIGGS_DEBUG") and step < 14:
                    logger.info(
                        f"[dbg] step {step:2d} pos={pos} row={nxt.tolist()} "
                        f"nd={num_delay} nr={num_rem} fin={finished}"
                    )
                if finished:
                    break
                # Early stop: the model loops on an identical row when it has run out
                # of content but failed to emit EOS (the silent tail). Drop the run.
                if prev_row is not None and torch.equal(nxt, prev_row):
                    repeat += 1
                    if repeat >= silence_patience:
                        del rows[-repeat:]  # drop the degenerate repeated run, keep first instance
                        logger.info(f"early stop at step {step}: dropped {repeat} repeated rows (no EOS)")
                        break
                else:
                    repeat = 0
                prev_row = nxt
                if use_trace:
                    tr["feed_tokens"](nxt)
                else:
                    cur_in = nxt
                pos += 1
        finally:
            if use_trace:
                ttnn.release_trace(self.mesh_device, tr["trace_id"])
        decode_s = time.perf_counter() - _t_dec

        # Steady-state decode throughput excludes the one-time prefill + trace build.
        self._last_timing = {
            "prefill_s": prefill_s,
            "trace_build_s": trace_build_s,
            "decode_s": decode_s,
            "rows": len(rows),
            "decode_tok_per_s": (len(rows) / decode_s) if decode_s > 0 else 0.0,
        }
        logger.info(
            f"timing: prefill {prefill_s*1e3:.0f}ms  trace-build {trace_build_s*1e3:.0f}ms  "
            f"decode {decode_s:.2f}s ({self._last_timing['decode_tok_per_s']:.1f} tok/s over {len(rows)} rows)"
        )

        if not finished:
            logger.warning(f"no natural EOS (max_new_tokens / degeneration); appending EOS row")
            rows.append(torch.full((self.K,), self.cfg.audio_stream_eos_id, dtype=torch.long))

        # [1, T, K] with leading all-BOS row, as batch_decode expects.
        audio_seq = torch.stack([bos_row] + rows, dim=0).unsqueeze(0)
        logger.info(f"generated {len(rows)} audio rows (finished={finished}) -> audio_seq {tuple(audio_seq.shape)}")
        return audio_seq

    # ---- batched (multi-stream) decode ---------------------------------------
    def _embed_rows(self, rows):
        """Host embed-and-sum for a batch of per-user frames.

        ``rows``: list of B tensors, each ``[K]`` codebook ids for one user's
        current frame. Returns the summed hidden as a torch bf16 tensor
        ``[1, 1, B, dim]`` to stream into the batched decode trace. Mirrors
        ``decode_step_audio``'s host embed-sum (the path the accuracy gate
        validated), so the traced forward is a pure blocks+norm+head pass.
        """
        K, cb = self.K, self.cb_size
        table = self.model._audio_embed_host_table()  # [audio_vocab, dim], float
        offsets = torch.arange(K, dtype=torch.long) * cb
        vecs = [table[(r.long().clamp(0, cb - 1) + offsets)].sum(dim=0) for r in rows]  # each [dim]
        h = torch.stack(vecs, dim=0)  # [len(rows), dim]
        # Pad batch to a full tile (32). The decode residual add (h + attn_out) requires the
        # hidden's logical batch to match the attention output, which tt-metal always pads to
        # 32; a sub-tile batch (2..31) triggers "invalid subtile broadcast" in that add. The
        # real streams occupy rows 0..N-1; rows N..31 are inert padding (current_pos/KV stay
        # at the real batch, so the pad rows never touch the cache).
        pad = ttnn.TILE_SIZE - h.shape[0]
        if pad > 0:
            h = torch.cat([h, torch.zeros(pad, h.shape[1], dtype=h.dtype)], dim=0)
        return h.view(1, 1, ttnn.TILE_SIZE, -1).to(torch.bfloat16)

    def _build_decode_trace_batch(self, start_positions):
        """Capture a batched audio decode step (B streams in the tile-height slot).

        Same device forward as ``_build_decode_trace`` but B users decode in
        lockstep: one ~2.17 GB weight read per replay serves all B streams — the
        DRAM-bandwidth amortization that batching buys. The summed audio embedding
        is host-computed and streamed in as ``[1,1,B,dim]`` each step
        (``feed_hidden``); per-user sampling / RAS / delay-pattern / ragged EOS all
        stay on host between replays. Reads back the full ``[1,1,B,K*cb]`` logits.
        """
        K, cb = self.K, self.cb_size
        dev = self.mesh_device
        B = len(start_positions)
        skip_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        norm_cfg = self.args.get_norm_config("lm_head", Mode.DECODE, None)
        cluster_shape = self.args.cluster_shape

        # Hidden is padded to a full tile (32 rows) so the residual add matches the
        # attention output's batch; only the first B rows are real (see _embed_rows).
        h_dev = ttnn.from_torch(
            torch.zeros(1, 1, ttnn.TILE_SIZE, self.args.dim, dtype=torch.bfloat16),
            device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )
        cp_dev = ttnn.from_torch(
            torch.tensor(start_positions, dtype=torch.int32), device=dev, dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=cluster_shape),
        )
        rope_idxs = self.rope_setup.get_rot_idxs(torch.tensor(start_positions, dtype=torch.int32), on_host=False)

        def step_body():
            h = ttnn.to_memory_config(h_dev, skip_mem_cfg)
            rot_mats = self.rope_setup.get_rot_mats(rope_idxs)
            for blk in self.model.layers:
                h = blk(h, cp_dev, rot_mats, mode=Mode.DECODE, is_audio_token=True)
            h = self.model.norm(h, Mode.DECODE, norm_config=norm_cfg)
            lg = self.model.audio_lm_head(h)
            return ttnn.slice(lg, (0, 0, 0, 0), (1, 1, B, K * cb))

        def advance_pos(positions):
            cp_host = ttnn.from_torch(
                torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=cluster_shape),
            )
            ttnn.copy_host_to_device_tensor(cp_host, cp_dev)
            ridx_host = self.rope_setup.get_rot_idxs(torch.tensor(positions, dtype=torch.int32), on_host=True)
            ttnn.copy_host_to_device_tensor(ridx_host, rope_idxs)

        def feed_hidden(h_torch):
            h_host = ttnn.from_torch(
                h_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(dev)
            )
            ttnn.copy_host_to_device_tensor(h_host, h_dev)

        advance_pos(start_positions)
        _ = step_body()
        ttnn.synchronize_device(dev)
        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        logits_out = step_body()
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)
        return {
            "trace_id": trace_id, "logits_out": logits_out, "B": B,
            "advance_pos": advance_pos, "feed_hidden": feed_hidden,
        }

    def _read_logits_batch(self, logits_out, B):
        K, cb = self.K, self.cb_size
        t = ttnn.to_torch(logits_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        return t.reshape(-1)[: B * K * cb].reshape(B, K, cb).float()

    @torch.no_grad()
    def generate_batch(
        self,
        conversations,
        max_new_tokens=750,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        seed=1234,
        silence_patience=32,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
    ):
        """Batched multi-stream generation: B genuinely different prompts decoded
        in lockstep on one chip, sharing a single weight read per step.

        Each conversation is prefilled into its own KV-cache row (``user_id=b``),
        then a batched decode trace runs with per-user positions. Sampling, RAS,
        the delay-pattern state machine and EOS all run per-user on host, so
        streams terminate independently (ragged EOS): a finished stream is frozen
        (its position held, output discarded) while the rest keep decoding until
        all finish or ``max_new_tokens``.

        Returns ``(audio_seqs, timing)`` — ``audio_seqs`` is a list of B tensors,
        each ``[1, T_b, K]`` in delay-pattern format ready for ``batch_decode``.

        The decode is wired to exactly ``max_batch_size`` streams (the rope/attention
        decode configs are sized for it), so fewer conversations are padded up with
        dummy streams that ride along and are dropped from the result.
        """
        B_real = len(conversations)
        N = self.max_batch_size
        assert B_real <= N, f"batch {B_real} > model max_batch_size {N}"
        K = self.K
        eos = self.cfg.audio_stream_eos_id
        if seed is not None:
            torch.manual_seed(seed)

        # Pad to exactly N streams (decode is fixed to max_batch_size); dummies are dropped.
        convs = list(conversations) + [conversations[-1]] * (N - B_real)
        input_ids_list = []
        for conv in convs:
            ids, audio_in, _ = self.build_inputs(conv)
            assert audio_in is None, "generate_batch supports text-only prompts (TTS); voice-clone batching TBD"
            input_ids_list.append(ids)
        S_list = [int(ids.shape[0]) for ids in input_ids_list]

        _t_pre = time.perf_counter()
        for b, ids in enumerate(input_ids_list):
            self.model.prefill_text(ids, self.rope_setup, audio_token_id=self.processor.audio_token_id, user_id=b)
        prefill_s = time.perf_counter() - _t_pre
        logger.info(f"batched prefill: N={N} (real={B_real}) prompt_lens={S_list} in {prefill_s*1e3:.0f}ms")

        bos_row = torch.full((K,), self.cfg.audio_stream_bos_id, dtype=torch.long)
        eos_row = torch.full((K,), eos, dtype=torch.long)
        states = [initialize_delay_pattern_state(bos_row.view(K, 1), self.cfg) for _ in range(N)]

        _t_tr = time.perf_counter()
        tr = self._build_decode_trace_batch(S_list)
        trace_build_s = time.perf_counter() - _t_tr

        rows_per_user = [[] for _ in range(N)]
        finished = [False] * N
        prev_row = [None] * N
        repeat = [0] * N
        positions = list(S_list)
        tr["feed_hidden"](self._embed_rows([bos_row.clone() for _ in range(N)]))

        _t_dec = time.perf_counter()
        steps_run = 0
        try:
            for _step in range(max_new_tokens):
                tr["advance_pos"](positions)
                ttnn.execute_trace(self.mesh_device, tr["trace_id"], cq_id=0, blocking=True)
                ttnn.synchronize_device(self.mesh_device)
                logits_all = self._read_logits_batch(tr["logits_out"], N)
                steps_run += 1
                next_rows = []
                for b in range(N):
                    if finished[b]:
                        next_rows.append(eos_row)
                        continue
                    lg = logits_all[b]  # [K, cb]
                    sel = self._sample_tokens(lg, temperature, top_k, top_p)
                    sel = self._ras(sel, lg, rows_per_user[b], ras_win_len, ras_win_max_num_repeat)
                    nd, nr = states[b]
                    nxt, _active, nd, nr, fin = apply_delay_pattern_to_selected_audio_tokens(sel, self.cfg, nd, nr)
                    states[b] = (nd, nr)
                    nxt = nxt.to(torch.long)
                    rows_per_user[b].append(nxt)
                    if fin:
                        finished[b] = True
                    elif prev_row[b] is not None and torch.equal(nxt, prev_row[b]):
                        repeat[b] += 1
                        if repeat[b] >= silence_patience:
                            del rows_per_user[b][-repeat[b] :]
                            finished[b] = True
                    else:
                        repeat[b] = 0
                    prev_row[b] = nxt
                    positions[b] += 1
                    next_rows.append(eos_row if finished[b] else nxt)
                # Stop when the REAL streams are all done (dummies just ride along).
                if all(finished[b] for b in range(B_real)):
                    break
                tr["feed_hidden"](self._embed_rows(next_rows))
        finally:
            ttnn.release_trace(self.mesh_device, tr["trace_id"])
        decode_s = time.perf_counter() - _t_dec

        audio_seqs = []
        for b in range(B_real):
            rows = rows_per_user[b] + ([] if finished[b] else [eos_row])
            audio_seqs.append(torch.stack([bos_row] + rows, dim=0).unsqueeze(0))

        total_useful = sum(len(rows_per_user[b]) for b in range(B_real))
        timing = {
            "B": B_real,
            "N": N,
            "prefill_s": prefill_s,
            "trace_build_s": trace_build_s,
            "decode_s": decode_s,
            "steps_run": steps_run,
            "rows_per_user": [len(rows_per_user[b]) for b in range(B_real)],
            "per_stream_tok_per_s": steps_run / decode_s if decode_s > 0 else 0.0,
            "device_tok_per_s": N * steps_run / decode_s if decode_s > 0 else 0.0,
            "useful_tok_per_s": total_useful / decode_s if decode_s > 0 else 0.0,
        }
        self._last_batch_timing = timing
        logger.info(
            f"batched decode: N={N} (real={B_real}) steps={steps_run} rows/user={timing['rows_per_user']} "
            f"decode {decode_s:.2f}s | per-stream {timing['per_stream_tok_per_s']:.1f} tok/s  "
            f"device-aggregate {timing['device_tok_per_s']:.1f} tok/s"
        )
        return audio_seqs, timing

    # ---- waveform output -----------------------------------------------------
    @staticmethod
    def _trim_silence(wf, sr=SAMPLING_RATE, rel_thresh=0.02, abs_thresh=0.004, pad_ms=120):
        """Trim leading/trailing near-silence from a 1D waveform.

        Threshold is the max of an absolute floor and a fraction of the clip's
        peak frame energy, so quiet clips aren't over-trimmed. A small pad is
        kept around the speech so it doesn't sound clipped.
        """
        x = wf.detach().float().flatten()
        fr = int(0.02 * sr)
        if x.numel() < 2 * fr:
            return wf
        nb = x.numel() // fr
        e = x[: nb * fr].reshape(nb, fr).pow(2).mean(-1).clamp_min(1e-12).sqrt()
        thr = max(abs_thresh, rel_thresh * float(e.max()))
        active = (e > thr).nonzero().flatten()
        if active.numel() == 0:
            return wf
        pad = int(pad_ms / 1000 * sr)
        start = max(0, int(active[0]) * fr - pad)
        end = min(x.numel(), (int(active[-1]) + 1) * fr + pad)
        return wf[start:end]

    def to_waveforms(self, audio_seq, trim=True):
        """audio_seq [1,T,K] -> list of 1D waveform tensors.

        Default uses the HF codec. With HIGGS_TTNN_CODEC=1, runs the ported
        TTNN DacDecoder (tt/codec.py) for the token->waveform step instead,
        replicating batch_decode's BOS/EOS trim + delay-pattern revert on host.
        """
        if os.environ.get("HIGGS_TTNN_CODEC") == "1":
            wfs = [self._ttnn_codec_decode(audio_seq)]
        else:
            wfs = self.processor.batch_decode(audio_seq)
        if trim:
            wfs = [self._trim_silence(w) for w in wfs]
        return wfs

    def _ttnn_codec_decode(self, audio_seq):
        from models.demos.audio.higgs_audio_v2.tt.codec import tt_decode

        p = self.processor
        bos, eos = p.audio_stream_bos_id, p.audio_stream_eos_id
        ids = audio_seq
        start = int((ids == bos).all(-1).nonzero()[-1, -1])
        ids = ids[:, start:]
        eos_rows = (ids == eos).all(-1).nonzero()
        end = int(eos_rows[eos_rows[:, 0] == 0, 1].min()) if (eos_rows[:, 0] == 0).any() else ids.shape[1]
        codes = p.revert_delay_pattern(ids[0, 1:end]).clip(0, bos - 1).transpose(0, 1).unsqueeze(0)  # [1,K,T]
        if not hasattr(self, "_ttdec"):
            from models.demos.audio.higgs_audio_v2.tt.codec import TtDacDecoder, TtRvqDequant

            self._ttdec = TtDacDecoder(self.mesh_device, p.audio_tokenizer.acoustic_decoder)
            self._rvq = TtRvqDequant(self.mesh_device, p.audio_tokenizer.quantizer, p.audio_tokenizer.fc2)
        wf = tt_decode(self.mesh_device, p.audio_tokenizer, codes, ttdec=self._ttdec, rvq=self._rvq)
        return wf.squeeze().detach().cpu()

    def save(self, audio_seq, out_path):
        waveforms = self.to_waveforms(audio_seq)
        self.processor.save_audio(waveforms, str(out_path), sampling_rate=SAMPLING_RATE)
        wf = waveforms[0]
        dur = wf.numel() / SAMPLING_RATE
        logger.info(f"saved {out_path}  ({dur:.2f}s, {wf.numel()} samples @ {SAMPLING_RATE}Hz)")
        return out_path, dur
