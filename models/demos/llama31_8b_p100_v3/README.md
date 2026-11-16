# Llama-3.1-8B on Blackhole P100

Llama-3.1-8B for a single Blackhole P100. The model is TT's `models/ttt_compat` Llama with the layers that run on dedicated
`ttnn.experimental.llama31_8b_v3` ops swapped in through subclasses; everything else (weights, tokenizer, generator, demo flow)
is imported unchanged.

Decode uses the EAGLE-3 draft head for Llama-3.1-8B-Instruct (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` from Hugging Face, or a
local copy given by `HF_EAGLE_HEAD`); the generated tokens are the same as `models/ttt_compat` greedy decode on the same card.

```
pytest models/demos/llama31_8b_p100_v3/demo/demo.py -k "performance-ci-1"
pytest models/demos/llama31_8b_p100_v3/tests/test_matches_main.py
```

Only Llama-3.1-8B on one P100 (7 DRAM banks) takes these ops (`tt/p100.py`); any other card, mesh or model runs the
`models/ttt_compat` path unchanged. Each op's wiring is one module, `tt/ops_<op>.py`, reached only through `tt/wiring.py`:
without that module (or without the op built) the layer makes the original TT call in its place.

```
pytest models/demos/llama31_8b_p100_v3/tests/test_wiring.py
```
