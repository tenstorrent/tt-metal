# K3.1 golden converter (local only)

One-shot producer of the K3.1 golden (`chunked_group_a_v1` vLLM trace -> K3's `per_stream_safetensors_v1`).
Removed from PR #64830 by tt_metal_tracker-8v1.7 (user rule: producer/diagnostic tools are not part of the PR).
`convert_golden.py` is byte-identical to tt-metal `models/demos/deepseek_v3_d_p/tests/kimi_k3/convert_golden.py`
at 050ce1431f9 (sha256 72b5b2a6...). `test_convert_golden.py` is its round-trip test, extracted from
`test_k31_golden.py` at the same commit (full original kept as `test_k31_golden.at-050ce1431f9.py`).
Both import tt-metal modules: run from a tt-metal checkout root with PYTHONPATH=$PWD.

    python /home/mvasilijevic/repos/k31-local-tools/golden-converter/convert_golden.py SRC DST [--layers N]
