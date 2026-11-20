# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview decode on the SC1 layout: token ids in, logits out, over a cache a prefill left
(tenstorrent/tt-metal#65893), for the first layers or all 78.

    prompt --prefill (the prefill stack, tt/prefill_bridge.py)--> KV / index-key caches
           --bridge--> TtHy4KvCache --decode steps (this module)--> logits --> next token

TtHy4Model is the decode model, WeightCache its converted weights on disk, Hy4Engine the whole
sequence on one open mesh (prefill, hand-over, free the prefill weights, load the decode weights,
decode). What all 78 layers give against HF and what they cost is at the end of this text.

One decode step, for a batch of users (one new token each), composes the modules that exist:

    tokens [U]            TtHy4Embedding                    [1, 1, U, hidden/4] fp32, users down the rows
    4 streams             ihc_expand                        [1, 1, U, 4 * hidden/4]
    per layer             TtHy4DecoderLayer                 iHC pre / post, both norms, dense FFN or MoE:
                                                            all users in one call
      its attention       TtHy4AttentionDecode              new token's rows appended to the cache,
                                                            top-2048 selection (full layers) or the
                                                            full layer's picks (shared layers), row
                                                            gather, attention: ONE USER PER CALL
    head readout          TtIHCHead                         [1, 1, U, hidden/4]
    final norm, LM head   TtHy4LMHead                       [1, 1, U, vocab/32] per chip, fp32
    next token            TtHy4LMHead.greedy / top_k        on the device; ttnn.sampling is not used

Continuing after a prefill. The prefill stack hands over caches and no state of the last position
(tt/prefill_bridge.py), so the first step re-runs the LAST PROMPT TOKEN: for a prompt of N tokens,
step 0 is ``decode_step([ids[N - 1]], [N - 1])``. It rewrites row N - 1 of every layer with decode's
own values, attends over positions [0, N - 1] and predicts token N; step j feeds token N - 1 + j.
The caller owns every user's position: the caches keep no length.

Users are rows. Everything outside attention takes the users as the rows of one tensor, which is
how the modules were built (T tokens down the tile rows). The attention sublayer does not:
TtHy4AttentionDecode takes one user's token, one user's stripes and one position per call, so a
layer slices each user's row out, runs the users one after the other and concatenates the
outputs. That loop is the gap between this model and the target (8 users x 1M context at 75
tokens per second per user is one step of the whole batch in 13.3 ms, 0.17 ms per layer): it
needs the indexer's score over a users batch dim, cache.locate with a row offset per user, the
row collect on [1, U, 2048, 576] and the attention core with users on the batch dim
(tt/attention_decode.py lists the same). Each user has its own TtHy4IndexCache, because a shared
layer must find its own user's picks. Rows of a batch are not bit-identical: two users with the
same prompt at the same position come out one fp32 unit apart in the streams within a step or
two (first seen after an iHC post-mix), and since the MoE router is a discrete choice that can
end in a different expert pick and logits a few tenths apart (measured; the next token was the
same on every step checked).

What still goes through the host on every token, each with what would replace it:
    begin_decode_step, per user     cos / sin of the position (2 uploads), the cache's append indices
                                    (a host table and an upload), the query position (an upload).
                                    On the device: position x inverse frequencies and ttnn.cos / sin,
                                    kv_cache.locate_rows on a position tensor, a counter. See
                                    tt/attention_decode.py.
    the token id                    none when the token is the device's own (``greedy`` output fed to
                                    ``decode_step``: TtHy4Embedding.table_rows). A token id handed in
                                    from the host (the first step, a forced token) is one upload.
    the users loop                  Python, U iterations per layer (above).
    the context length              a host int per user; the selection's shapes follow it (a new
                                    program every 4096 positions). A traced step has to score the
                                    whole stripe.
Nothing is read back to the host inside a step unless ``record`` or ``profile`` is given.

Weights. A layer's tensors are read from the checkpoint and converted inside the modules'
constructors: one MoE layer is 39 GB in fp32 on the host and takes 20 to 33 s. WeightCache keeps
what the constructors upload on disk, so that is paid once: 79 groups (the embedding and head,
78 layers), written a few layers per device job (tests/test_model_full.py
test_build_weight_cache). Measured on the 8x4 Galaxy, BFP8 weights with BFP4 routed experts:

    build, from the checkpoint, with the write    2569 s in all over 8 jobs: 26 to 56 s per MoE layer
                                                  (33 s mean), 3.3 s the dense layer, 21 s the
                                                  embedding and head
    on disk                                       469.0 GB: 6.04 GB per MoE layer (6.06 with an
                                                  indexer), 0.67 GB the dense layer, 2.97 GB the
                                                  embedding and head
    DRAM per chip while resident                  15.14 GB of 34.18: 0.195 GB per MoE layer (0.196
                                                  with an indexer), 0.025 GB the dense layer,
                                                  0.094 GB the embedding and head
    load, from the cache                          1.7 s per layer with the files in the page cache
                                                  (12 layers in 20 s), 2.7 to 3.3 s when they come
                                                  off the disk: 78 layers in 210 to 259 s
    BFP8 routed experts (not the plan)            one MoE layer measured: 0.347 GB of DRAM per chip
                                                  (0.196 with BFP4) and 10.89 GB on disk (6.06),
                                                  built in 46 s. For the 77 MoE layers: 26.8 GB per
                                                  chip in all, 841 GB on disk; it fits a chip only
                                                  with the prefill weights gone, and leaves 7 GB
                                                  for every cache

A cached group is built from placeholders of its large tensors (PLACEHOLDERS), so a load reads
only the small ones from the checkpoint. The same steps, scores and tokens come out of a model
loaded from the cache as out of one converted from the checkpoint (six layers: the lowest PCC of
every stage, of the readout and of the logits equal to six digits, the same tokens).

The prefill stack's weights are a second set: 20.90 GB per chip for 78 layers with its storage
(its own measurement: 21.1). Both do not fit a chip together (36.0 of 34.18 GB before any
cache), so a 78-layer run is prefill -> PrefillBridge.release_model (0.4 s, 0.03 GB left) ->
load the decode weights -> decode; Hy4Engine is that sequence. Serving needs one set of weights
for both.

Precision is what the modules default to: BFP8 attention / shared-expert / dense-FFN weights,
BFP4 routed experts, BF16 sublayer activations, fp32 streams, norms and logits.

What a step costs today
-----------------------
Measured on the 8x4 Galaxy with checkpoint weights, untraced, the ops issued back to back
(tests/test_model_full.py, tests/test_model_prefill_decode.py; ``layer_times``). Nothing here is
optimised and nothing says what the kernels take.

    one user, ms per layer        dense FFN + full indexer   MoE + full indexer   MoE + shared indexer
    6 layers, context 5.1K        25.7                       28.3                 20.4
    78 layers, context 5.1K       25.4                       27.9                 20.3
    78 layers, context 8.2K       25.7                       28.3                 20.5
    78 layers, context 32.8K      26.0                       28.6                 20.6
    78 layers, context 131K       27.4                       30.1                 20.6
    eight users, 6 layers, 1K and 5.1K
                                  127                        130                  66.5
    eight users, 78 layers, 1K and 5.1K
                                  127.7                      128.2                66.1

  A step of all 78 layers for one user: 1.75 to 1.78 s at 1K, 5K, 8K and 33K of context and 1.82 s
  at 131K (0.55 to 0.57 tokens per second, 22.3 to 23.1 ms per layer on average, 131x to 135x the
  budget below; what grows with the context is the selection of a full layer, 2.7 ms at 1K, 3.4
  at 8K, 9.0 at 131K), with the
  embedding (1.4 ms) and the head with greedy (4.5 ms); 1.89 s per step measured over 31 greedy
  steps of a short chat prompt with the tokens fed back on the device. For a batch of eight users
  (four at 1K, four at 5.1K): 6.52 s per step, 0.153 tokens per second per user and 1.23 in all,
  82.8 ms per layer on average, 484x the budget. By the pieces of a layer
  (a synchronize after each, an idle one taken off), the same at every depth: iHC 8.3 to 8.9, the
  two norms 2.0 to 2.6, MoE 3.7 to 4.5 (dense FFN 1.5 to 1.7), whatever the number of users;
  attention 15.8 to 16.9 (full) and 7.1 to 7.4 (shared) for one user, 128 to 132 and 56 for
  eight; inside a full layer's attention, one user: new token 5.8 to 6.1, selection 2.7 to 3.4,
  locate 1.1 to 1.3, row gather 0.1 to 0.4, collect 0.8 to 1.1, core 4.6 to 5.4.
* The target is 0.171 ms per layer for the batch of 8 users (13.3 ms for a step of 78 layers).
  A shared layer is 119 times that for one user and 389 times for eight; a full layer 166 and 759
  times.
* Everything but attention already takes the users as rows and costs the same for 8 as for 1.
  Attention is 8 calls. With attention batched too, and costing what one user costs now, a step
  would still be about 1.8 s: the time is the host issuing ops (tt/attention_decode.py measures
  about 80 us per ttnn call on 32 chips, which makes a shared layer's 20 ms some 250 calls; iHC
  alone, two sites of slices, mixes and one all-reduce each, is 8.5 ms for one token). That
  needs a traced step, and then fused ops for iHC and for the indexed attention
  (tt/attention_decode.py).
* The first token of a prompt waits for much more than a step, because the two weight sets take
  turns on the chips: 364 to 446 s at 78 layers = build the prefill runtime from its weight cache
  (122 to 166 s) + compile it (9 to 21 s) + prefill (1.3 s for up to 4096 tokens, 2.4 to 2.6 s for
  two chunks, 10.0 s for 32768 tokens, 40.6 s for 131072: 3230 to 3420 tokens per second from
  8192 tokens up) + hand the caches over (7.5 to 7.8 s up to 5K, 15 to 16 s at 8K, 41 s at 32K,
  141 s at 131K: the latents go through the host) + free the prefill weights (0.4 to 1.9 s) +
  load the decode weights (181 to 259 s) + the first step (4 to 10 s, it compiles its programs).
  At 131072 tokens that is 473 s.
* L1_SMALL: with the prefill stack's runtime built on the same mesh, the steps of one user leave
  10,112 bytes per core in use, at 6 layers and at 78, and those of 1 and 8 users together 14,848
  (every distinct collective program keeps its semaphores there). The mesh is opened with 32,768
  (prefill_bridge.SHARED_L1_SMALL): more than the stack's own 1,536, and than the 8,192 the
  decode-step tests open their torus mesh with.

All 78 layers against HF
------------------------
tests/test_model_full.py, BFP8 weights, BFP4 routed experts, a BFP8 decode cache filled from the
prefill stack's scaled-FP8 storage. HF is the real hy_v4 model in BF16, cut to the same depth.

* Each layer on HF's own input (``layer_inputs``; 78 layers, 6 decode steps of a 1022-token
  prompt). Attending over HF's cache rows: attention output 0.9933 or better on every layer
  (median 0.9992), streams after the attention site 0.9994, MoE input 0.9988, streams after the
  layer 0.9871 at layer 39 and 0.99 or better on the other 77 (median 0.9993). Attending over the
  rows the prefill stack wrote: attention output 0.9671 (layer 73; 59 layers at 0.99 or better),
  streams after the layer 0.9811 (76 layers at 0.99 or better). One token's MoE output through
  BFP4 experts is 0.91 to 0.999, lower where the router picks one or two experts other than HF's.
  No layer is broken at any depth; what follows is what 78 of them add up to.
* The device on its own streams, lowest PCC over the layers and 6 teacher-forced steps of a
  1022-token prompt, with the lowest k_pass of the rows it was handed (over the layers):
      layers   streams after a layer   logits   handed-over k_pass   next token = HF's (decisive steps)
      6        0.9965                  0.9959   0.998                6 of 6 (3 of 3)
      12       0.9943                  0.9909   0.991                4 of 6 (2 of 2)
      24       0.9877                  0.9876   0.978                3 of 6 (3 of 3)
      48       0.9661                  0.9868   0.964                5 of 6 (3 of 3)
      78       0.53                    0.945    0.917                3 of 6 (2 of 2)
  (78 layers over the two decisive steps alone: 0.885 and 0.971.) A step is decisive when HF's
  top-1 leads by a logit or more; below that HF's cached generation and its own full pass do not
  always pick the same token in these goldens.
* HF against itself gives the scale of that: two BF16 runs of the same 48 layers that differ only
  in the tokens BEHIND the scored position agree on that position's streams to 0.9962 at layer 30
  (tests/test_model_full_host.py): one BF16 step on 126 of the 6144 values of layer 5's attention
  output is 0.4% of PCC twenty-five layers on, in HF itself. On a step where the model is
  undecided (HF's margin 0.05) the device's deep layers come out as another computation
  altogether (streams 0.53 at layer 69, logits 0.945).
* What decode inherits. The rows the prefill stack hands over, against HF's own cache at 78
  layers (1021 rows per layer, its scaled-FP8 storage): k_pass 0.917 at its lowest (layer 73),
  median over the layers 0.978, 22 layers at 0.99 or better; k_rope 0.996 or better; index keys
  0.9947 or better. The same decode steps over those rows and over HF's rows written in their
  place, lowest over the layers and the 6 steps (streams / logits / share of HF's top-2048 picks):
      1022-token prompt   the stack's rows, scaled FP8   0.53  / 0.945 / all     HF's rows 0.80  / 0.967 / all
      5118-token prompt   the stack's rows, scaled FP8   0.23  / 0.888 / 54%     HF's rows 0.935 / 0.982 / 92.5%
                          the stack's rows, BF16_RM      0.929 / 0.980 / 90.7%
  (at 1022 tokens every position is picked). With the stack's BF16_RM storage (1152 instead of
  704 bytes per token per layer, converted to this BFP8 cache on the device in 0.65 s) the
  logits of every one of the six steps are within 0.01 of what HF's own rows give; with its
  scaled-FP8 storage every step is lower (logits 0.970 to 0.990 on five of them against 0.982 to
  0.993) and the sixth, on which HF is undecided, falls apart (logits 0.888 against 0.980,
  streams 0.23, 54% of the picks at layer 53). The handed-over k_pass at 5118 tokens is 0.942
  at its lowest and 0.988 at the median layer from BF16_RM. The cached latent of the last
  prompt position, which both stacks compute, is as close to HF's from the
  decode step (median over the layers 0.9935 and 0.982 for the two prompts, lowest 0.916 and
  0.820) as from the prefill stack (0.992 and 0.981, 0.920 and 0.826). So the decode steps are as
  accurate as the prefill, the rows they are handed limit a 78-layer run, and of those rows it is
  the FP8 packing of the latent that costs most: this module's own BFP8 cache holds the same
  latent in 612 bytes with seven mantissa bits where the packed row spends 704 on three.
* Generation. A chat prompt (31 tokens through the checkpoint's template), 32 greedy tokens with
  the device feeding its own tokens back: all 32 are HF's greedy tokens ('We need answer user
  asks: "Explain in three sentences why the sky is blue." ...'). Teacher-forced on HF's tokens:
  top-1 equal on 31 of 32 steps and on all 31 decisive ones (the other step has an HF margin of
  0.2, and HF's cached generation picks the device's token there); logits PCC 0.894 and 0.895 on
  the first two steps, 0.93 to 0.99 after them, mean 0.968.
* Long prompts against the 78-layer fp32 reference of the prefill stack's golden tooling, the
  last position (the first decode step). 32768 tokens (LongBench): logits 0.984, the next token
  is the reference's (its margin 0.42); the cached latent of that position is 0.941 or better
  on every layer from the decode step (median 0.986) and 0.937 from the prefill stack (0.985).
  8192 tokens (a document cut mid-line): logits 0.945; the reference is undecided there (margin
  0.006) and the device takes its second token; streams 0.98 or better through layer 29 and 0.43
  at layer 69, the cached latent of the decode step down to 0.21 at layer 73 where the prefill
  stack's own is 0.79 at its lowest: an undecided position again. 131072 tokens: prefill,
  hand-over and decode run (99 stripes of 4352 rows per chip, 15.37 GB per chip with the decode
  weights); there is no reference at that length and nothing of it is checked.
"""

from __future__ import annotations

import gc
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference import decoder as reference
from models.experimental.hy4_preview.reference.attention import rope_pairing
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.reference.indexer import full_layers as indexer_layer_types
from models.experimental.hy4_preview.reference.lm_head import LMHeadConfig
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.attention_decode import TtHy4AttentionDecode, begin_decode_step
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import (
    TtHy4DecoderLayer,
    attention_weights,
    hidden_to_torch,
    streams_to_device,
    streams_to_torch,
)
from models.experimental.hy4_preview.tt.embedding import TtHy4Embedding
from models.experimental.hy4_preview.tt.ihc import TtIHCHead, ihc_expand
from models.experimental.hy4_preview.tt.indexer import TtHy4IndexCache, TtHy4Indexer
from models.experimental.hy4_preview.tt.kv_cache import TtHy4KvCache
from models.experimental.hy4_preview.tt.lm_head import TtHy4LMHead
from models.experimental.hy4_preview.tt.moe import EXPERT_WEIGHT_DTYPE

# What ``record`` keeps of every layer, read back to the host.
RECORDED = ("attn_in", "attn_out", "streams_mid", "mlp_in", "mlp_out", "streams_out")
# The pieces a layer's time is split into when ``profile`` is given.
LAYER_STAGES = ("ihc", "norm", "attention", "mlp")


@dataclass
class ModelConfig:
    """Defaults are tencent/Hy4-preview. ``num_layers`` cuts the decoder to its first layers (the
    head, the final norm and the LM head are then applied to the cut stack, as the goldens do)."""

    num_layers: int = 78
    decoder: reference.DecoderConfig = field(default_factory=reference.DecoderConfig)
    indexer: IndexerConfig = field(default_factory=IndexerConfig)
    head: LMHeadConfig = field(default_factory=LMHeadConfig)
    indexer_types: tuple | None = None

    def __post_init__(self):
        if self.indexer_types is None:
            self.indexer_types = tuple(indexer_layer_types(self.num_layers))
        assert len(self.indexer_types) == self.num_layers, (len(self.indexer_types), self.num_layers)

    def is_full(self, layer: int) -> bool:
        return self.indexer_types[layer] == "full"


# Where the converted decode weights are cached when the caller names no directory.
WEIGHT_CACHE_ENV = "HY4_DECODE_WEIGHT_CACHE"
# The tensors whose host conversion is the build time: the fused routed experts (39 GB per MoE layer
# in fp32) and the two vocab tables. A group that is in the weight cache is built from placeholders
# of them: shapes without data, which the modules' reshapes and transposes accept as they are.
PLACEHOLDERS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj", "model.embed_tokens.weight", "lm_head.weight")


class Checkpoint:
    """Tensors by name out of a BF16 safetensors checkpoint directory, as fp32."""

    def __init__(self, root):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        assert index.is_file(), f"{index} not found"
        self.weight_map = json.loads(index.read_text())["weight_map"]

    def tensors(self, names, placeholders: bool = False) -> list:
        """``placeholders``: the PLACEHOLDERS tensors come back as shapes without data (torch's
        "meta" device), read from the shard's header only."""
        import safetensors

        out = []
        for name in names:
            with safetensors.safe_open(str(self.root / self.weight_map[name]), framework="pt") as f:
                if placeholders and name.endswith(PLACEHOLDERS):
                    out.append(torch.empty(f.get_slice(name).get_shape(), dtype=torch.float32, device="meta"))
                else:
                    out.append(f.get_tensor(name).float())
        return out

    def layer(self, layer: int, dense: bool, placeholders: bool = False) -> dict:
        """One layer's tensors by their name under ``model.layers.{layer}.``."""
        names = reference.LAYER_PARAMETERS[dense]
        return dict(zip(names, self.tensors(reference.checkpoint_names(layer, dense), placeholders)))

    def indexer(self, layer: int) -> dict:
        keys = list(Hy4Indexer.CHECKPOINT_NAMES.values())
        return dict(zip(keys, self.tensors([f"model.layers.{layer}.self_attn.indexer.{k}" for k in keys])))


def _dtype_name(dtype) -> str:
    return str(dtype).split(".")[-1].lower()


class WeightCache:
    """The device tensors a model build uploads, kept on disk in the form they are uploaded in.

    The decode modules convert their weights inside their constructors, with ttnn.from_torch on
    the mesh. Inside ``group(name)`` every such upload is numbered and goes through a file,
    ``<name>.<number>.tensorbin`` (the converted tensor of all chips, ttnn.dump_tensor): written
    on the first build, loaded with ttnn.load_tensor after that without the torch tensor being
    read. A group is one build unit ("embedding_and_head", "layer_17"); it counts as cached once
    its stamp ``<name>.json`` is written, which happens when the whole group has been built, so
    a build that died halfway is redone. The stamp lists every upload (shape, dtype, layout);
    a cached group whose constructors ask for anything else fails loudly instead of loading
    another tensor's file.

    The directory name carries what the files depend on: the mesh, the hidden axis, both weight
    dtypes, the RoPE pairing of the process (reference/attention.py rope_pairing: the frames of
    tt/frames.py reorder the rope rows of q_b, kv_a and the indexer by it, with the same shapes
    either way, so a cache of the other pairing would load without an error and be wrong) and
    FORMAT, which has to be raised whenever a module changes what it uploads or in which order
    (the cache cannot see that).

    The numbering lives in a patch of ``ttnn.from_torch`` that is in place only inside ``group``:
    the modules have no hook for their uploads (tt/common.py MeshPlan would be the place for one).
    """

    FORMAT = 1

    def __init__(self, root, plan: MeshPlan, weight_dtype=ttnn.bfloat8_b, expert_dtype=None):
        if expert_dtype is None:  # TtHy4MoE's policy
            expert_dtype = EXPERT_WEIGHT_DTYPE if weight_dtype == ttnn.bfloat8_b else weight_dtype
        shape = "x".join(str(n) for n in (plan.mesh_shape or (1, 1)))
        self.directory = Path(root) / (
            f"{shape}_hidden{plan.hidden_axis}_{_dtype_name(weight_dtype)}_experts-{_dtype_name(expert_dtype)}"
            f"_rope-{rope_pairing()}_v{self.FORMAT}"
        )
        self._active = None

    def _stamp(self, name: str) -> Path:
        return self.directory / f"{name}.json"

    def complete(self, name: str) -> bool:
        return self._stamp(name).is_file()

    def stamp(self, name: str) -> dict:
        return json.loads(self._stamp(name).read_text())

    def bytes_on_disk(self, name: str | None = None) -> int:
        pattern = "*.tensorbin" if name is None else f"{name}.*.tensorbin"
        return sum(path.stat().st_size for path in self.directory.glob(pattern))

    @contextmanager
    def group(self, name: str):
        """Yields whether the group is loaded from the cache (True) or built and written (False)."""
        assert self._active is None, f"group {self._active} is still open"
        cached = self.complete(name)
        listed = self.stamp(name)["tensors"] if cached else []
        seen = []
        original = ttnn.from_torch
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)

        def upload(tensor, *args, **kwargs):
            device = kwargs.get("device")
            if device is None or args:
                return original(tensor, *args, **kwargs)
            index = len(seen)
            entry = {
                "shape": list(tensor.shape),
                "dtype": _dtype_name(kwargs.get("dtype")),
                "layout": _dtype_name(kwargs.get("layout")),
            }
            seen.append(entry)
            path = directory / f"{name}.{index:03d}.tensorbin"
            if cached:
                assert index < len(listed) and listed[index] == entry, (
                    f"weight cache {path}: the build asks for {entry}, the cache holds "
                    f"{listed[index] if index < len(listed) else 'nothing'}; raise WeightCache.FORMAT or delete the group"
                )
                return ttnn.load_tensor(path, device=device)
            host = original(tensor, **{k: v for k, v in kwargs.items() if k not in ("device", "memory_config")})
            scratch = directory / f"{name}.{index:03d}.{os.getpid()}.tmp.tensorbin"
            try:
                ttnn.dump_tensor(scratch, host)
                os.replace(scratch, path)
            finally:
                scratch.unlink(missing_ok=True)
            return host.to(device, kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG)

        self._active = name
        ttnn.from_torch = upload
        t0 = time.perf_counter()
        try:
            yield cached
        finally:
            ttnn.from_torch = original
            self._active = None
        if cached:
            assert len(seen) == len(listed), f"weight cache group {name}: {len(seen)} uploads, {len(listed)} cached"
        else:
            stamp = {"format": self.FORMAT, "tensors": seen, "build_seconds": time.perf_counter() - t0}
            scratch = self.directory / f"{name}.{os.getpid()}.tmp"
            scratch.write_text(json.dumps(stamp))
            os.replace(scratch, self._stamp(name))


def weight_cache_for(plan: MeshPlan, weight_cache, weight_dtype, expert_dtype) -> WeightCache | None:
    """``weight_cache``: a WeightCache, a directory, True (the directory in $HY4_DECODE_WEIGHT_CACHE)
    or None / False (convert from the checkpoint every time)."""
    if weight_cache is None or weight_cache is False or isinstance(weight_cache, WeightCache):
        return weight_cache or None
    if weight_cache is True:
        weight_cache = os.environ.get(WEIGHT_CACHE_ENV)
        assert weight_cache, f"weight_cache=True needs ${WEIGHT_CACHE_ENV}"
    return WeightCache(weight_cache, plan, weight_dtype, expert_dtype)


@contextmanager
def _group(weights: WeightCache | None, name: str):
    if weights is None:
        yield False
    else:
        with weights.group(name) as cached:
            yield cached


def layer_group(layer: int) -> str:
    return f"layer_{layer}"


HEAD_GROUP = "embedding_and_head"


def build_head(plan: MeshPlan, cfg: ModelConfig, ckpt: Checkpoint, weights: WeightCache | None = None):
    """(TtHy4Embedding, TtIHCHead, TtHy4LMHead) from the checkpoint or the weight cache."""
    with _group(weights, HEAD_GROUP) as cached:
        (table,) = ckpt.tensors(["model.embed_tokens.weight"], placeholders=cached)
        embedding = TtHy4Embedding(plan, cfg.head, table)
        del table
        fn, base, scale, norm_weight, head = ckpt.tensors(
            [
                "model.hc_head.hc_head_fn",
                "model.hc_head.hc_head_base",
                "model.hc_head.hc_head_scale",
                "model.norm.weight",
                "lm_head.weight",
            ],
            placeholders=cached,
        )
        hc_head = TtIHCHead(plan, cfg.decoder.ihc, fn, base, scale)
        lm_head = TtHy4LMHead(plan, cfg.head, {"norm_weight": norm_weight, "weight": head})
        del head
    return embedding, hc_head, lm_head


def build_layer(
    plan: MeshPlan,
    cfg: ModelConfig,
    ckpt: Checkpoint,
    cache: TtHy4KvCache,
    layer: int,
    index_caches: list,
    weights: WeightCache | None = None,
    weight_dtype=ttnn.bfloat8_b,
    expert_dtype=None,
    core: str = "absorb",
    head_sum: str = "fp32",
):
    """(TtHy4AttentionDecode, TtHy4DecoderLayer) of one layer, from the checkpoint or the weight cache."""
    d = cfg.decoder
    dense = d.is_dense(layer)
    with _group(weights, layer_group(layer)) as cached:
        tensors = ckpt.layer(layer, dense, placeholders=cached)
        attention = TtHy4Attention(plan, d.attention, attention_weights(tensors), weight_dtype=weight_dtype)
        indexer = None
        if cfg.is_full(layer):
            indexer = TtHy4Indexer(plan, cfg.indexer, ckpt.indexer(layer), layout=cache.layout, head_sum=head_sum)
        decode = TtHy4AttentionDecode(plan, attention, cache, layer, index_caches[0], indexer, core=core)
        decoder_layer = TtHy4DecoderLayer(
            plan,
            d,
            tensors,
            dense=dense,
            weight_dtype=weight_dtype,
            attention=_UsersAttention(decode, index_caches),
            expert_dtype=expert_dtype,
        )
        del tensors
    gc.collect()
    return decode, decoder_layer


def build_weight_cache(
    plan: MeshPlan,
    cfg: ModelConfig,
    checkpoint,
    weight_cache,
    layers=None,
    head: bool = True,
    weight_dtype=ttnn.bfloat8_b,
    expert_dtype=None,
    progress=None,
) -> dict:
    """Write the weight-cache groups of ``layers`` (default: every layer of ``cfg``) and, with
    ``head``, of the embedding and head, one group on the device at a time; a group that is
    already cached is skipped. Needs the mesh (the tensors are converted with its mesh mappers)
    but nothing of a model: each group is built, written and freed. Returns
    {group: {"built", "seconds", "bytes" on disk, "dram_gb" per chip while resident (when built)}}."""
    from models.experimental.hy4_preview.tt.kv_cache import KvCacheConfig

    ckpt = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
    weights = weight_cache_for(plan, weight_cache, weight_dtype, expert_dtype)
    assert weights is not None, "name the weight cache to build"
    layers = list(range(cfg.num_layers)) if layers is None else list(layers)
    full = tuple(layer for layer in range(cfg.num_layers) if cfg.is_full(layer))
    # the layers' decode modules want a cache to point at; a one-round one is a megabyte per chip
    cache = TtHy4KvCache(
        plan.mesh_device, KvCacheConfig(num_layers=cfg.num_layers, indexer_layers=full, max_context=4096), layers=layers
    )
    index_caches = [TtHy4IndexCache(list(cfg.indexer_types))]
    report = {}

    def allocated_gb():
        view = ttnn.get_memory_view(plan.mesh_device, ttnn.BufferType.DRAM)
        return view.total_bytes_allocated_per_bank * view.num_banks / 1e9

    def build(name, fn):
        built = not weights.complete(name)
        t0 = time.perf_counter()
        entry = {"built": built}
        if built:
            before = allocated_gb()
            modules = fn()
            ttnn.synchronize_device(plan.mesh_device)
            entry["dram_gb"] = allocated_gb() - before
            del modules
            gc.collect()
        report[name] = dict(entry, seconds=time.perf_counter() - t0, bytes=weights.bytes_on_disk(name))
        if progress is not None:
            progress(name, report[name])

    try:
        if head:
            build(HEAD_GROUP, lambda: build_head(plan, cfg, ckpt, weights))
        for layer in layers:
            build(
                layer_group(layer),
                lambda: build_layer(plan, cfg, ckpt, cache, layer, index_caches, weights, weight_dtype, expert_dtype),
            )
    finally:
        cache.deallocate()
    return report


class _UsersAttention:
    """The attention callable of a decoder layer in decode: rows are users, each run as its own step."""

    def __init__(self, decode: TtHy4AttentionDecode, index_caches: list):
        self.decode = decode
        self.index_caches = index_caches

    def __call__(self, x, steps):
        rows, width = x.shape[-2], x.shape[-1]
        assert rows == len(steps), f"{rows} rows for {len(steps)} steps"
        outs = []
        for row, step in enumerate(steps):
            xu = x if rows == 1 else ttnn.slice(x, [0, 0, row, 0], [1, 1, row + 1, width])
            self.decode.index_cache = self.index_caches[step.user]
            outs.append(self.decode(xu, step))
        return outs[0] if rows == 1 else ttnn.concat(outs, dim=2)


class TtHy4Model(LightweightModule):
    """Layers ``0 .. cfg.num_layers - 1`` in decode over ``cache``, with the embedding and the head.

    ``checkpoint`` is a Checkpoint (or a directory). ``cache`` is the TtHy4KvCache the steps read
    and append to; it must hold every layer and its ``max_users`` is the largest batch a step
    takes. ``core`` and ``head_sum`` are TtHy4AttentionDecode's and TtHy4Indexer's.
    ``seconds`` records the build: per layer and for the embedding / head. ``progress(name)`` is
    called after the embedding and head ("embedding_and_head") and after every layer ("layer L")
    have been uploaded, for a caller that measures memory as the model grows. ``weight_cache``
    (a directory, a WeightCache, or True for $HY4_DECODE_WEIGHT_CACHE) keeps the converted
    weights on disk: the first build writes them, later builds load them.
    """

    def __init__(
        self,
        plan: MeshPlan,
        cfg: ModelConfig,
        cache: TtHy4KvCache,
        checkpoint,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=None,
        core: str = "absorb",
        head_sum: str = "fp32",
        progress=None,
        weight_cache=None,
    ):
        self.plan, self.cfg, self.cache = plan, cfg, cache
        progress = progress or (lambda name: None)
        ckpt = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
        assert all(layer in cache.latent for layer in range(cfg.num_layers)), "the cache does not hold every layer"
        self.weights = weights = weight_cache_for(plan, weight_cache, weight_dtype, expert_dtype)
        self.seconds = {"layers": []}
        self.index_caches = [TtHy4IndexCache(list(cfg.indexer_types)) for _ in range(cache.cfg.max_users)]

        t0 = time.perf_counter()
        self.embedding, self.hc_head, self.lm_head = build_head(plan, cfg, ckpt, weights)
        self.seconds["embedding_and_head"] = time.perf_counter() - t0
        progress("embedding_and_head")

        self.layers, self.decodes = [], []
        for layer in range(cfg.num_layers):
            t0 = time.perf_counter()
            decode, decoder_layer = build_layer(
                plan, cfg, ckpt, cache, layer, self.index_caches, weights, weight_dtype, expert_dtype, core, head_sum
            )
            self.decodes.append(decode)
            self.layers.append(decoder_layer)
            self.seconds["layers"].append(time.perf_counter() - t0)
            progress(f"layer {layer}")

    # ------------------------------------------------------------------------------------------
    def _sync(self):
        ttnn.synchronize_device(self.plan.mesh_device)

    def _timed_layer(self, layer: TtHy4DecoderLayer, streams, steps, seconds: dict):
        """TtHy4DecoderLayer.forward with a device synchronize after every piece, the wall time of
        each added to ``seconds`` under its LAYER_STAGES name and the number of synchronizes it
        includes under "laps / <name>" (a synchronize costs time of its own; the caller measures
        an idle one and takes it off)."""
        last = [time.perf_counter()]

        def lap(name):
            self._sync()
            now = time.perf_counter()
            seconds[name] = seconds.get(name, 0.0) + now - last[0]
            seconds[f"laps / {name}"] = seconds.get(f"laps / {name}", 0) + 1
            last[0] = now

        self._sync()
        last[0] = time.perf_counter()
        h, post = layer.hc_attn.hc_pre(streams)
        lap("ihc")
        x = layer.input_layernorm(h)
        lap("norm")
        out = ttnn.typecast(layer.attention(x, steps=steps), layer.stream_dtype)
        lap("attention")
        mid = layer.hc_attn.hc_post(out, streams, post)
        h, post = layer.hc_mlp.hc_pre(mid)
        lap("ihc")
        x = layer.post_attention_layernorm(h)
        lap("norm")
        out = ttnn.typecast(layer.mlp(x), layer.stream_dtype)
        lap("mlp")
        streams = layer.hc_mlp.hc_post(out, mid, post)
        lap("ihc")
        return streams

    def decode_step(
        self,
        tokens,
        positions,
        users=None,
        record: list | None = None,
        profile: dict | None = None,
        layer_inputs: list | None = None,
    ):
        """One new token for each of ``users`` (default: every user of the cache).

        tokens: host ints / a torch tensor [U], or the device tensor ``greedy`` returned for the same
        users (uint32 [1, 1, 1, U]). positions: one int per user, where its token goes; the cache
        must hold that user's positions below it. Returns the logits, per chip [1, 1, U, vocab/32]
        fp32 (chip i holds token ids [i * 3776, (i + 1) * 3776)); ``lm_head.to_torch`` reads them.

        ``record``, a list, receives one dict per layer with the RECORDED tensors on the host
        ([U, ...]) and, on a full layer, ``picks``: per user (positions int64 [K], valid bool [K]).
        ``profile``, a dict, receives the wall times: "layers" (one dict per layer: LAYER_STAGES,
        "laps / <stage>" for the synchronizes each includes, and "attention / <stage>" for the
        tt/attention_decode.py STAGES inside the attention time, summed over the users, each of
        which ends in a synchronize too), "embedding", "head". Both add synchronizes and host
        readbacks that a plain step has none of; ``layer_times`` is the measurement without them.

        ``layer_inputs`` takes the layers apart for a test: one host tensor [U, n, hidden] per layer
        (a reference's streams going INTO that layer), uploaded and fed to the layer in place of
        the output of the layer below. The step still appends the rows and makes the selections in
        order, so with ``record`` every layer's own error shows on the reference's input, free of
        what the layers below it did. The head is applied to the last layer's output.
        """
        cache, plan, n = self.cache, self.plan, self.cfg.decoder.ihc.n
        assert record is None or profile is None, "record and profile do not combine: a timed layer keeps nothing"
        users = list(range(cache.cfg.max_users)) if users is None else list(users)
        assert len(positions) == len(users), f"{len(positions)} positions for {len(users)} users"
        if isinstance(tokens, ttnn.Tensor):
            assert tokens.shape[-1] == len(users), (tuple(tokens.shape), len(users))
            rows = self.embedding.table_rows(tokens)
        else:
            tokens = torch.as_tensor(tokens, dtype=torch.int64).flatten()
            assert tokens.numel() == len(users), f"{tokens.numel()} tokens for {len(users)} users"
            rows = self.embedding.host_table_rows(tokens)

        t0 = time.perf_counter()
        streams = ihc_expand(self.embedding(rows), n)
        if profile is not None:
            self._sync()
            profile["embedding"] = time.perf_counter() - t0
            profile["layers"] = []
        attention_cfg = self.cfg.decoder.attention
        steps = [
            begin_decode_step(plan, cache, attention_cfg, int(position), user=user, profile=profile is not None)
            for user, position in zip(users, positions)
        ]
        assert layer_inputs is None or len(layer_inputs) == len(self.layers), "one input per layer"
        for index, layer in enumerate(self.layers):
            rec = None if record is None else {}
            if layer_inputs is not None:
                streams = streams_to_device(plan, layer_inputs[index])
            if profile is None:
                streams = layer(streams, record=rec, steps=steps)
            else:
                before = [dict(step.seconds) for step in steps]
                seconds = {}
                streams = self._timed_layer(layer, streams, steps, seconds)
                for step, old in zip(steps, before):
                    for stage, total in step.seconds.items():
                        name = f"attention / {stage}"
                        seconds[name] = seconds.get(name, 0.0) + total - old.get(stage, 0.0)
                profile["layers"].append(seconds)
            if record is not None:
                host = {}
                for key in RECORDED:
                    if key.startswith("streams"):
                        host[key] = streams_to_torch(plan, rec[key], n)
                    else:
                        host[key] = hidden_to_torch(plan, rec[key])
                if self.cfg.is_full(index):
                    indexer = self.decodes[index].indexer
                    host["picks"] = []
                    for step in steps:
                        picked, valid = indexer.picked(self.index_caches[step.user].picks(index))
                        host["picks"].append((picked[0], valid[0]))
                record.append(host)

        t0 = time.perf_counter()
        readout = self.hc_head(streams)
        logits = self.lm_head(readout)
        if record is not None:
            record.append({"hc_head": hidden_to_torch(plan, readout)})
        if profile is not None:
            self._sync()
            profile["head"] = time.perf_counter() - t0
        for step in steps:
            step.end()
        return logits

    def layer_times(self, tokens, positions, users=None, repeats: int = 4) -> dict:
        """The wall time of every layer on its own, without a synchronize inside: one step in which
        each layer's forward is issued ``repeats`` times back to back on the same input (after one
        call that compiles its programs) between two synchronizes. A repeat rewrites the same cache
        rows and makes the same selection, so the step is left as a plain one would leave it.
        Returns {"embedding", "layers": [seconds per layer], "head"} (the head with ``greedy``)."""
        cache, plan, n = self.cache, self.plan, self.cfg.decoder.ihc.n
        users = list(range(cache.cfg.max_users)) if users is None else list(users)
        tokens = torch.as_tensor(tokens, dtype=torch.int64).flatten()

        def timed(fn):
            out = fn()
            self._sync()
            t0 = time.perf_counter()
            for _ in range(repeats):
                out = fn()
            self._sync()
            return out, (time.perf_counter() - t0) / repeats

        out = {"layers": []}
        streams, out["embedding"] = timed(lambda: ihc_expand(self.embedding(self.embedding.host_table_rows(tokens)), n))
        attention_cfg = self.cfg.decoder.attention
        steps = [begin_decode_step(plan, cache, attention_cfg, int(p), user=u) for u, p in zip(users, positions)]
        for layer in self.layers:
            streams, seconds = timed(lambda: layer(streams, steps=steps))
            out["layers"].append(seconds)
        _, out["head"] = timed(lambda: self.greedy(self.lm_head(self.hc_head(streams))))
        for step in steps:
            step.end()
        return out

    def greedy(self, logits):
        """The argmax token of every row, on the device: uint32 [1, 1, 1, U], the same on every
        chip. Feed it to the next ``decode_step`` as ``tokens``."""
        return self.lm_head.greedy(logits)

    def top_k(self, logits, k: int):
        """(values [1, 1, U, k], token ids uint32 [1, 1, U, k]), sorted, on the device."""
        return self.lm_head.top_k(logits, k)

    def tokens_to_torch(self, tokens) -> torch.Tensor:
        """``greedy``'s result on the host: int64 [U]."""
        return ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).flatten().long()

    def generate(self, tokens, positions, steps: int, users=None) -> torch.Tensor:
        """``steps`` greedy steps from ``tokens`` at ``positions``, every token after the first fed
        back on the device. Returns the generated ids on the host, int64 [steps, U] (read back once
        per step, after the next step has been issued the device tensor)."""
        positions = [int(p) for p in positions]
        out = []
        for step in range(steps):
            logits = self.decode_step(tokens, [p + step for p in positions], users=users)
            tokens = self.greedy(logits)
            out.append(tokens)
        return torch.stack([self.tokens_to_torch(t) for t in out])


# ---------------------------------------------------------------------------------------------
# Prompt in, tokens out
# ---------------------------------------------------------------------------------------------
class Hy4Engine:
    """Prefill by the prefill stack, hand-over, decode by TtHy4Model, on one open 8x4 mesh
    (tt/prefill_bridge.device_params is how it must have been opened).

    The order is the one all 78 layers need, because the two weight sets do not fit a chip
    together (module docstring): ``prefill`` builds the stack's runtime, prefills every prompt
    into its user slot, copies the rows into the decode cache and frees the stack's weights and
    storage; ``load`` then brings the decode weights in (from the weight cache when it is named);
    ``generate`` / ``step`` decode. One ``prefill`` call takes every user's prompt: a second
    round of prompts would have to free the decode weights first, which is not built.

    ``max_context`` is each user's cache length in positions (prompt + generated), rounded up to
    the 8192 the stack's storage is cut in. ``kv_format`` / ``handover`` are PrefillBridge's
    (``handover="device"`` moves on the device what the bridge can, ``"host"`` everything through
    the host). ``seconds`` and ``dram`` (GB allocated per chip) are filled as the stages run.
    """

    def __init__(
        self,
        mesh_device,
        num_layers: int = 78,
        max_users: int = 1,
        max_context: int = 8192,
        checkpoint=None,
        weight_cache=None,
        kv_format: str = "scaled_fp8",
        handover: str = "device",
        cache_dtype: str = "bfp8",
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=None,
        core: str = "absorb",
        head_sum: str = "fp32",
    ):
        from models.experimental.hy4_preview.tt import prefill_bridge
        from models.experimental.hy4_preview.tt.kv_cache import KvCacheConfig

        assert handover in ("device", "host"), handover
        self.mesh_device = mesh_device
        self.plan = MeshPlan(mesh_device)
        self.cfg = ModelConfig(num_layers=num_layers)
        self.max_users = max_users
        self.max_context = -(-max_context // (2 * prefill_bridge.CHUNK)) * (2 * prefill_bridge.CHUNK)
        self.checkpoint = Checkpoint(checkpoint or os.environ["HY4_HF_MODEL"])
        self.weight_cache = weight_cache
        self.handover = handover
        self.model_args = dict(weight_dtype=weight_dtype, expert_dtype=expert_dtype, core=core, head_sum=head_sum)
        self._bridge_module = prefill_bridge
        self.bridge = prefill_bridge.PrefillBridge(
            mesh_device, num_layers=num_layers, num_users=max_users, max_seq_len=self.max_context, kv_format=kv_format
        )
        full = tuple(layer for layer in range(num_layers) if self.cfg.is_full(layer))
        self.cache_cfg = KvCacheConfig(
            num_layers=num_layers,
            indexer_layers=full,
            max_users=max_users,
            max_context=self.max_context,
            dtype=cache_dtype,
        )
        self.cache = None
        self.model = None
        self.lengths = {}  # user -> the context length (the next free position)
        self.last_token = {}  # user -> the token decode feeds next
        self.pending = {}  # user -> True until the user's first step, which re-runs the last prompt token
        self.seconds = {}
        self.dram = {}

    def _mark(self, name: str):
        self.dram[name] = round(self._bridge_module.dram(self.mesh_device)["allocated_gb"], 4)

    # ------------------------------------------------------------------------------------------
    def prefill(self, prompts) -> dict:
        """``prompts``: {user: token ids} (or a list, user 0 first). Returns {user: N}."""
        assert self.cache is None, "one prefill round per engine"
        prompts = dict(enumerate(prompts)) if isinstance(prompts, (list, tuple)) else dict(prompts)
        bridge = self.bridge
        self._mark("open")
        bridge.build()
        self._mark("prefill_weights_and_storage")
        for user, ids in prompts.items():
            ids = [int(t) for t in (ids.tolist() if isinstance(ids, torch.Tensor) else ids)]
            assert len(ids) >= 2, "a prompt needs two tokens: decode re-runs the last one over the rest"
            assert len(ids) < self.max_context, f"a prompt of {len(ids)} tokens leaves no room in {self.max_context}"
            self.lengths[user] = bridge.prefill(ids, user=user)
            self.last_token[user] = ids[-1]
            self.pending[user] = True
        self._mark("prefilled")
        self.cache = TtHy4KvCache(self.mesh_device, self.cache_cfg)
        self._mark("with_decode_cache")
        t0 = time.perf_counter()
        host = None
        on_device = self.handover == "device"
        for user in prompts:
            if host is None and not (on_device and bridge.device_latents()):
                host = bridge.read_host()
            bridge.hand_over(self.cache, user, host=host, on_device=on_device)
        del host
        ttnn.synchronize_device(self.mesh_device)
        self.seconds["handover"] = time.perf_counter() - t0
        bridge.release_model()
        self._mark("prefill_model_released")
        bridge.release_caches()
        gc.collect()
        self._mark("prefill_storage_released")
        for key in ("build_runtime", "compile", "prefill", "release_model"):
            self.seconds[f"prefill / {key}"] = bridge.seconds.get(key)
        return dict(self.lengths)

    def load(self, progress=None) -> TtHy4Model:
        """The decode weights, after ``prefill`` has freed the stack's."""
        assert self.cache is not None, "prefill() first: the decode cache is allocated there"
        assert self.model is None, "already loaded"
        t0 = time.perf_counter()
        self.model = TtHy4Model(
            self.plan,
            self.cfg,
            self.cache,
            self.checkpoint,
            progress=progress,
            weight_cache=self.weight_cache,
            **self.model_args,
        )
        ttnn.synchronize_device(self.mesh_device)
        self.seconds["load_decode_weights"] = time.perf_counter() - t0
        self._mark("decode_weights_and_cache")
        return self.model

    # ------------------------------------------------------------------------------------------
    def next_inputs(self, users=None):
        """(tokens, positions) of the next step of ``users``: a user fresh out of prefill re-runs its
        last prompt token at N - 1 (the stack hands over no state), any other feeds its last token
        at N."""
        users = sorted(self.lengths) if users is None else list(users)
        tokens = [self.last_token[u] for u in users]
        positions = [self.lengths[u] - 1 if self.pending[u] else self.lengths[u] for u in users]
        return tokens, positions

    def step(self, users=None, tokens=None, record=None, profile=None, layer_inputs=None):
        """One decode step for ``users`` (default: every prefilled user). ``tokens`` replaces what
        is fed (teacher forcing; host ints). Returns the logits on the device; the caller
        reports each user's chosen token with ``accept``. ``record``, ``profile`` and
        ``layer_inputs`` are TtHy4Model.decode_step's."""
        users = sorted(self.lengths) if users is None else list(users)
        fed, positions = self.next_inputs(users)
        tokens = fed if tokens is None else [int(t) for t in tokens]
        assert max(positions) < self.max_context, f"a context has reached the cache's {self.max_context} positions"
        logits = self.model.decode_step(
            tokens, positions, users=users, record=record, profile=profile, layer_inputs=layer_inputs
        )
        for user, position in zip(users, positions):
            self.lengths[user] = position + 1
            self.pending[user] = False
        return logits

    def accept(self, tokens, users=None):
        """The tokens the next step feeds (host ints, one per user)."""
        users = sorted(self.lengths) if users is None else list(users)
        for user, token in zip(users, tokens):
            self.last_token[user] = int(token)

    def generate(self, steps: int, users=None) -> torch.Tensor:
        """``steps`` greedy tokens for every user, each fed back on the device. Returns int64
        [steps, U] on the host; the ids are read back once, after the last step was issued."""
        users = sorted(self.lengths) if users is None else list(users)
        tokens, positions = self.next_inputs(users)
        assert max(positions) + steps <= self.max_context, f"{steps} steps do not fit the cache"
        out = self.model.generate(tokens, positions, steps, users=users)
        for user, position, token in zip(users, positions, out[-1].tolist()):
            self.lengths[user] = position + steps
            self.pending[user] = False
            self.last_token[user] = int(token)
        return out

    def rewind(self, user: int, length: int, last_token: int):
        """Take ``user`` back to a context of ``length`` positions whose last token is
        ``last_token``; the next step re-runs that token at ``length - 1``. Rows above stay in the
        cache and are never selected (a step attends below its own position)."""
        assert 2 <= length <= self.lengths[user], (length, self.lengths[user])
        self.lengths[user], self.last_token[user], self.pending[user] = length, int(last_token), True

    def close(self):
        """Free what the prefill stack still holds. Call it in a ``finally`` before the mesh closes."""
        self.bridge.close()
