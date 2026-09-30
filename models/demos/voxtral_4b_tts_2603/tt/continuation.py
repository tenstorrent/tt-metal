# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 of the `mistralai/Voxtral-4B-TTS-2603` end-to-end package: TEXT CONTINUATION.

The task is the causal LM's next-token prediction, TEACHER-FORCED over the real text (`score`):
one forward, and at every position the model's continuation logits and its greedy pick. It is NOT
run as a free-running greedy decode, because on this checkpoint that decode never terminates --
the untrained tied text head emitted eos (id 2) on 0/32 rows in 448 CPU steps and collapses into a
1-5 token repetition loop by step ~9, so the model's only stop rule never fires and any horizon
would be one this package invented. `generate` is kept for the section test only.

The checkpoint's text half is a `MistralForCausalLM` (26 layers, dim 3072, head_dim 128
explicitly, 32 query heads over 8 KV heads, hidden 9216, `rope_theta` 1e6, `norm_eps` 1e-5,
vocab 131072, tied embeddings, `sliding_window` None). Source B graduated a whole-stack port of
it -- the embedding, the 26 layers and the final RMSNorm -- twice, once as `encoder_stack` and
once as `mistral_model`, plus the LM head as `decoder_head` bound to `lm_head`. Greedy causal-LM
continuation is the task those three bodies implement, and it is the ONLY consumer of
`decoder_head`: the model's TTS chain reads its next token from the acoustic transformer's
semantic head, never from `lm_head`.

Three things this module is careful about, each of which was a real trap:

* **`encoder_stack` and `mistral_model` are BYTE-IDENTICAL aliases** -- the same graduated work
  product recorded twice, once under the sibling-registry tag and once under the model's own
  class name (`common.alias_pairs()` detects the pair rather than listing it). Running a sample
  through both would be the same arithmetic twice, so the real work is SPLIT BY BATCH ROW: at
  every step rows `[0:split]` go through `encoder_stack` and rows `[split:B]` through
  `mistral_model`, and the two halves are concatenated before `decoder_head`. Every sample is
  processed exactly once, and both bodies are covered by the per-sample PCC gate because that
  gate compares each of the 32 samples to its own golden.

* **The leading axis is the batch, and the bound is read off the tensor.** Both whole-stack stubs
  originally reshaped to `[1, 1, seq, dim]` and returned `[1, seq, dim]` -- a hardcoded leading 1
  that drops samples 1..31. The stubs were fixed to take `batch = input_ids.shape[0]`, and this
  module RE-VERIFIES the bound on every forward rather than trusting it: each half's output has
  to come back with its own row count, and the two halves have to reassemble to B.

* **Causality comes from SDPA's `is_causal`, so the batch has to stay UNPADDED.**
  `common.build_batch_inputs()` gives 32 prompts each truncated to exactly 32 real tokens, so
  every row is genuine content and the implied mask is the plain lower-triangular one. If a
  ragged batch ever needs to be driven here, it needs a real additive `[B, 1, S, S]` mask in the
  stub. `forward_logits` therefore REJECTS an `attention_mask` outright instead of accepting and
  ignoring one: an ignored mask is indistinguishable from a respected one until the numbers are
  wrong, and the stubs do not consume it.

COHERENCE CAVEAT: this is a TTS checkpoint. The backbone emits AUDIO codebook tokens and its tied
TEXT head is effectively untrained, so the continuation text is near-uniform garbage even when the
load is bit-correct. That does not weaken anything -- the gate compares TT against the HF reference
on the SAME input, and garbage-that-matches is a valid parity result. The load is verified
STRUCTURALLY by the reference loader, never by reading the generated text.

No device is ever opened here; the caller owns the device.
"""
from __future__ import annotations

import ttnn

from . import common

# The two whole-stack bodies, in the order the batch split feeds them, and the head.
BODY_STUBS = ("encoder_stack", "mistral_model")
HEAD_STUB = "decoder_head"

STUBS = BODY_STUBS + (HEAD_STUB,)

# Full depth of `model.layers`, read from the reference at build time -- never written here.
# The floor is 3: the structural stack walk needs >= 3 same-typed members, so a cap below that
# hides the very stack the cap exists to size.
MIN_LAYERS = 3


def _tile_ceil(value: int) -> int:
    return -(-int(value) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def resolve_layer_cap(requested, full_depth: int, floor: int = MIN_LAYERS):
    """``(n_layers, note)`` for a `layers=` request against a stack of `full_depth`.

    `None` means EVERY layer -- never 0, which a builder reads as a zero-layer model. A request
    below `floor` is clamped UP and the clamp is reported, never applied silently: a 1- or
    2-layer stack is no longer a stack the structural walk can see, so silently honouring it
    would hide the thing the knob exists to size.
    """
    if requested is None:
        return full_depth, f"layers=None -> all {full_depth} layers"
    n = int(requested)
    if n <= 0:
        return floor, (
            f"layers={requested} is not a depth; clamped UP to the {floor}-layer floor "
            f"(>= {floor} same-typed members are what makes `model.layers` a walkable stack)"
        )
    if n < floor:
        return floor, (
            f"layers={n} is below the {floor}-layer floor; clamped UP to {floor} "
            f"(a stack shorter than {floor} is invisible to the structural walk)"
        )
    if n > full_depth:
        return full_depth, f"layers={n} exceeds the stack's {full_depth} layers; capped to {full_depth}"
    return n, f"layers={n} of {full_depth}"


class _CappedLayers:
    """Build-time only: present `model.layers` as its first `n` entries, then put it back.

    The graduated whole-stack stub compiles one callable per element of `model.layers`, so the
    depth knob is applied by handing `build()` a shorter list. This is shape/weight PREP at build
    time -- nothing here runs in the forward path, and the reference model is restored intact so
    the golden side is unaffected.
    """

    def __init__(self, model, n: int) -> None:
        self._model = model
        self._n = n
        self._saved = None

    def __enter__(self):
        import torch

        layers = self._model.layers
        if self._n >= len(layers):
            return self._model
        self._saved = layers
        self._model.layers = torch.nn.ModuleList(list(layers)[: self._n])
        return self._model

    def __exit__(self, *exc):
        if self._saved is not None:
            self._model.layers = self._saved
            self._saved = None
        return False


class Continuation:
    """Greedy causal-LM continuation on the two whole-stack bodies plus the LM head.

    Attributes
    ----------
    n_layers
        Layers actually built into each body (the `layers=` knob, after clamping).
    split_row
        Batch row at which `encoder_stack` hands over to `mistral_model`.
    vocab_size, hidden_size, full_layers
        Read off the reference config at build time.
    last_stop_reason
        Why the most recent `generate()` stopped -- `"horizon"` or `"eos"`.
    """

    def __init__(self, device, body_a, body_b, head, *, n_layers, full_layers, vocab_size, hidden_size, split_row=None):
        self.device = device
        self._body_a = body_a
        self._body_b = body_b
        self._head = head
        self.n_layers = int(n_layers)
        self.full_layers = int(full_layers)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self._split_row = split_row
        self.last_stop_reason = None
        self.stub_names = STUBS

    # ------------------------------------------------------------------ batch split
    def split_for(self, batch: int) -> int:
        """Row at which the two alias bodies divide `batch` samples.

        Defaults to `batch // 2`. `batch < 2` has no split to make, so one body takes
        everything -- returning 0 here would hand `encoder_stack` an empty slice.
        """
        if self._split_row is not None:
            split = int(self._split_row)
            if not 0 < split < batch:
                if batch < 2:
                    return batch
                raise ValueError(f"split_row={split} must lie strictly inside the batch [1, {batch - 1}]")
            return split
        return batch if batch < 2 else batch // 2

    # ------------------------------------------------------------------ forward
    def _stack_hidden(self, input_ids_tt, attention_mask=None, last_only=True):
        """`[B, S]` uint32 token ids -> `([B, L, hidden], batch, real_len)` off the two alias bodies.

        `L` is 1 (the last real position) when `last_only`, else every real position `S`.

        `attention_mask` is REJECTED, not ignored. The whole-stack stubs take causality from
        SDPA's `is_causal`, which is exact for an UNPADDED batch and silently wrong for a padded
        one; accepting a mask here would promise masking that nothing applies.
        """
        if attention_mask is not None:
            raise NotImplementedError(
                "the whole-stack stubs take causality from SDPA's is_causal and consume no mask, "
                "so a padded/ragged batch needs a real additive [B, 1, S, S] mask added to the "
                "stub first; build_batch_inputs() keeps every row genuine content instead"
            )
        shape = list(input_ids_tt.shape)
        if len(shape) != 2:
            raise ValueError(f"input_ids must be [B, S]; got {tuple(shape)}")
        batch, real_len = int(shape[0]), int(shape[1])
        if real_len < 1:
            raise ValueError("input_ids must carry at least one token")
        # A TILE-ALIGNED SEQUENCE, OR THE ANSWER IS SILENTLY ZERO. The whole-stack body returns
        # zeros past the last full tile, the head turns that into a flat logit row, and `argmax`
        # reports token 0 for every sample -- green wiring, garbage text. Padding is invisible to
        # the answer because the body's causality is SDPA's `is_causal`: the real positions cannot
        # see the tail, and only rows `< real_len` are ever read back.
        seq = _tile_ceil(real_len)
        if seq != real_len:
            input_ids_tt = ttnn.pad(input_ids_tt, [(0, 0), (0, seq - real_len)], value=0)
        split = self.split_for(batch)
        lo_pos = real_len - 1 if last_only else 0

        halves = []
        for body, lo, hi in self._assignments(input_ids_tt, batch, seq, split):
            rows = hi - lo
            ids = input_ids_tt if (lo == 0 and hi == batch) else ttnn.slice(input_ids_tt, [lo, 0], [hi, seq])
            hidden = body(ids)
            got = list(hidden.shape)
            # RE-VERIFY the batch bound rather than trust it: a body that reshaped to a hardcoded
            # leading 1 comes back one row wide, which is exactly how samples 1..B-1 disappear.
            if len(got) != 3 or int(got[0]) != rows or int(got[1]) != seq:
                raise AssertionError(
                    f"whole-stack body returned {tuple(got)} for {rows} rows x {seq} tokens -- "
                    "the leading axis is the BATCH and must be read off the input tensor"
                )
            # REAL positions only -- the tile pad above is zero rows.
            halves.append(ttnn.slice(hidden, [0, lo_pos, 0], [rows, real_len, self.hidden_size]))

        out = halves[0] if len(halves) == 1 else ttnn.concat(halves, dim=0)
        if int(out.shape[0]) != batch:
            raise AssertionError(f"batch split reassembled to {int(out.shape[0])} rows, expected {batch}")
        return out, batch, real_len

    def _head_rows(self, hidden, rows: int):
        """`[B, L, hidden]` -> `[rows, vocab]` logits, `rows = B * L`, through `decoder_head` ONCE.

        The head is POSITIONWISE, so every (sample, position) row is presented as one position of
        a single batch: the same arithmetic, and the 131072-wide weight is streamed once instead of
        once per sample. The hop is done in ROW_MAJOR, where the reshape is a contiguous view.
        """
        rm = ttnn.reshape(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), [1, rows, self.hidden_size])
        logits = self._head(ttnn.to_layout(rm, ttnn.TILE_LAYOUT))
        out = list(logits.shape)
        if int(out[-2]) != rows or int(out[-1]) != self.vocab_size:
            raise AssertionError(f"decoder_head returned {tuple(out)}, expected [..., {rows}, {self.vocab_size}]")
        return ttnn.reshape(logits, [rows, self.vocab_size])

    def forward_logits(self, input_ids_tt, attention_mask=None):
        """`[B, S]` uint32 token ids -> `[B, vocab]` logits for the LAST position."""
        last, batch, _ = self._stack_hidden(input_ids_tt, attention_mask, last_only=True)
        return self._head_rows(last, batch)

    def score(self, input_ids_tt, attention_mask=None):
        """Teacher-forced causal-LM forward: `[B, S]` ids -> `(next [B, S] uint32, logits [B, S, vocab])`.

        ONE forward over the real text: position `s` of `logits` is the model's next-token
        distribution given tokens `0..s`, and `next[:, s]` its greedy pick, chosen with
        `ttnn.argmax` ON DEVICE. Every position of every sample goes through the stack and the
        LM head -- not only the last.
        """
        hidden, batch, real_len = self._stack_hidden(input_ids_tt, attention_mask, last_only=False)
        rows = batch * real_len
        logits = self._head_rows(hidden, rows)
        # ROW_MAJOR first: the last-dim argmax is multi-core on ROW_MAJOR and SINGLE-core on TILE.
        picks = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        return (
            ttnn.reshape(picks, [batch, real_len]),
            ttnn.reshape(logits, [batch, real_len, self.vocab_size]),
        )

    def _assignments(self, input_ids_tt, batch: int, seq: int, split: int):
        """`(body, lo, hi)` row ranges. Both alias bodies run unless the batch cannot be split."""
        if batch < 2 or split >= batch:
            return [(self._body_a, 0, batch)]
        return [(self._body_a, 0, split), (self._body_b, split, batch)]

    # ------------------------------------------------------------------ generate
    def generate(self, input_ids_tt, horizon: int, eos_id=None):
        """Greedy continuation. Returns `(tokens [B, steps] uint32, step_logits)`.

        `step_logits` is a list of `[B, vocab]` tensors, one per step. The next token is chosen
        with `ttnn.argmax` ON DEVICE and concatenated onto the resident context ON DEVICE with
        `ttnn.concat` -- the token never round-trips through the host, and the context is never
        rebuilt with `ttnn.from_torch`.

        `eos_id` stops the loop once EVERY row has emitted it. That decision is a single scalar
        (a device-side `eq` -> `logical_or` accumulator -> `min` over the rows), not the tokens:
        one 1-element readback per step decides loop control, and the tokens themselves stay on
        device.
        """
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        batch = int(input_ids_tt.shape[0])

        context = input_ids_tt
        tokens, step_logits = [], []
        done = None
        self.last_stop_reason = "horizon"

        for _ in range(int(horizon)):
            logits = self.forward_logits(context)
            step_logits.append(logits)
            # ROW_MAJOR first: the last-dim argmax is multi-core on ROW_MAJOR and SINGLE-core on
            # TILE, and 32 x 131072 on one core is minutes of wall time for the same answer.
            next_token = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
            tokens.append(next_token)
            context = ttnn.concat([context, next_token], dim=-1)

            if eos_id is not None:
                hit = ttnn.eq(ttnn.typecast(ttnn.to_layout(next_token, ttnn.TILE_LAYOUT), ttnn.float32), float(eos_id))
                done = hit if done is None else ttnn.logical_or(done, hit)
                if float(ttnn.to_torch(ttnn.min(done)).flatten()[0]) >= 1.0:
                    self.last_stop_reason = "eos"
                    break

        out = tokens[0] if len(tokens) == 1 else ttnn.concat(tokens, dim=-1)
        if int(out.shape[0]) != batch:
            raise AssertionError(f"generate produced {int(out.shape[0])} rows, expected {batch}")
        self.context = context
        return out, step_logits


def build_continuation(device, hf_model, layers=None, counter=None) -> Continuation:
    """Build Call 2's resident bodies on `device`.

    `hf_model` is `common.load_reference_model()`. Weights come from `hf_model.model` for the two
    whole-stack bodies and from `hf_model.lm_head` for the head. Both alias bodies are built and
    both stay resident: the batch split needs them together (~6.9 GB each at bfloat16 weights,
    13.8 GB of the 29.2 GB registered usable at TP=1, plus ~0.8 GB for the head).

    `layers=N` caps `model.layers`; `None` is every layer. A capped build is still a MODEL -- the
    embedding, the final norm and the LM head all stay intact.
    """
    model = hf_model.model
    full_layers = len(model.layers)
    n_layers, note = resolve_layer_cap(layers, full_layers)
    print(f"[continuation] {note}", flush=True)

    with _CappedLayers(model, n_layers) as capped:
        built = {name: common.build_stub(name, device, capped, counter) for name in BODY_STUBS}
    head = common.build_stub(HEAD_STUB, device, hf_model.lm_head, counter)

    cont = Continuation(
        device,
        built[BODY_STUBS[0]],
        built[BODY_STUBS[1]],
        head,
        n_layers=n_layers,
        full_layers=full_layers,
        vocab_size=int(hf_model.config.vocab_size),
        hidden_size=int(hf_model.config.hidden_size),
    )
    print(
        f"[continuation] built {len(BODY_STUBS)} whole-stack bodies {BODY_STUBS} + {HEAD_STUB}; "
        f"n_layers={cont.n_layers}/{full_layers} hidden={cont.hidden_size} vocab={cont.vocab_size}",
        flush=True,
    )
    return cont
