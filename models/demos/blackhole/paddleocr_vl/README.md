# PaddleOCR-VL on Blackhole (P150)

[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) (0.9B
document-OCR VLM, Apache 2.0) end to end on a single Blackhole die, served over
the OpenAI-compatible API via vLLM.

## Supported HW

| Device | Mesh |
|--------|------|
| **P150** | 1x1 |

No multi-chip or Wormhole support exists for this model today.

## Measured accuracy

- **Text decoder** (ERNIE-4.5-0.3B, run unmodified through `models/tt_transformers`
  since it's Llama-shaped GQA): logits PCC 0.989 vs HuggingFace, matching top-1
  and 5/5 top-5 overlap on a held-out prompt.
- **Vision tower + end-to-end OCR** (direct path, 17 images spanning every
  bucket): character error rate 0.11% against the HuggingFace CPU reference; 16
  of 17 samples are exact matches.
- **Served endpoint** (vLLM, OpenAI-compatible API, one image per bucket): 0.00%
  CER, 1.7-4.9s per page, `finish_reason: "stop"`, identical answer on repeat.
  Once the reference corpus reached 20 images (5 per bucket), the served
  endpoint read all 20 at 0.00% CER.

## Measured performance

Streamed, so time-to-first-token and decode rate are measured separately, the
way a caller experiences them:

| bucket | image tokens | TTFT | decode |
|-------:|-------------:|-----:|-------:|
| 1024 | 256 | 85.9ms | 171.7 tok/s |
| 2048 | 512 | 131.6ms | 189.3 tok/s |
| 4096 | 1024 | 270.4ms | 187.0 tok/s |
| 6144 | 1280 | 425.1ms | 188.7 tok/s |

Gate: TTFT <= 600ms on the largest bucket, decode >= 150 tok/s/user. Both pass.
The decode win comes from `allow_force_argmax` (greedy is the only mode OCR
uses): 90 -> 188 tok/s/user, a 2.1x improvement. On-device sampling was tried
and is deliberately not claimed -- it collided with trace capture and produced
garbage at batch 1; `supports_sample_on_device` is `False` with that measurement
recorded next to it in `tt/generator_vllm.py`.

Vision-bucket compiles are warmed before trace capture (see
`tests/serve_hardening.py`'s docstring for why: a compile landing while a trace
is parked corrupts it silently rather than erroring). 66 requests over 4 randomized
passes plus 50 sequential showed no drift, median 0.25s.

## The measured resolution ceiling

The checkpoint's default `max_pixels` (1003520, ~100 DPI on A4) costs real
accuracy on small print: a dense 9pt-text page measured 4.98% CER there, and
0.07% at 1204224 -- exactly the top bucket (6144 patches / 1280 image tokens).
**The deployment serves at 1204224.**

Higher resolutions were built and tested, then removed. At 1605632 and above,
one printed page in five transcribed itself correctly and then transcribed
itself a second time; a dense page at 3211264 looped a phrase 141 times and ran
to the token limit, emitting 27140 characters for an 8228-character page. The
vision tower is not at fault -- short text at a 16384-patch grid comes back
exact -- so the fault is the decoder's stop behaviour once the image exceeds the
resolution the model was trained at. `tests/test_resolution_ceiling.py` guards
against silently re-widening the bucket table past this point.

Reaching genuinely higher DPI therefore means tiling the page into pieces that
each stay inside this envelope, not sending one larger image.

## Running the tests

The PCC and accuracy suites need `demo/golden/` (HuggingFace CPU reference
tensors and decoded text). It's gitignored -- ~94MB, reproducible on any machine
from the committed corpus -- so regenerate it first:

```
python models/demos/blackhole/paddleocr_vl/tests/generate_goldens.py
```

Then, on a P150 box:

```
MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
pytest models/demos/blackhole/paddleocr_vl/tests/ -s
```

Without `demo/golden/` staged, the PCC/accuracy/resolution-ceiling tests skip
cleanly (collection still succeeds) rather than failing; `test_vision_permutation.py`
needs no device or golden data and always runs. Set
`PADDLEOCR_VL_REQUIRE_ARTIFACTS=1` on an unattended run (e.g. CI) to turn those
skips into failures instead, so a staging break is caught rather than reported
as a green skip.

`tests/serve_smoke.py`, `tests/serve_hardening.py` and `tests/serve_perf.py` are
HTTP clients against an already-running vLLM server (`python -m ...
--served-model-name PaddlePaddle/PaddleOCR-VL-1.6` etc.) rather than pytest
cases; each exits non-zero on a gate miss.

CI pipeline registration (`tests/pipeline_reorg/*.yaml`, tiering, time budget) is
a deliberate follow-up, not part of this bring-up -- see the PR description.
