# Llama Tensor Prefetcher MPFE tuning

Run a single-card Blackhole smoke test with the full Llama 3.2 3B model:

```bash
MESH_DEVICE=P150 HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --use_prefetcher True \
  --repeat_batches 1 \
  --max_seq_len 1024 \
  --batch_size 1 \
  --max_generated_tokens 200 \
  --paged_attention True \
  --stop_at_eos 0 \
  --enable_trace \
  --mode full
```

Wrap the same unchanged model command with the generic MPFE tuner:

```bash
MESH_DEVICE=P150 HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
python3 tests/scripts/single_card/run_bh_tensor_prefetcher_mpfe_tune.py \
  --output-dir generated/llama-3.2-3b-tensor-prefetcher-mpfe \
  --search-runs 3 \
  --rank-runs 10 \
  --top-k 3 \
  --timeout-seconds 3600 \
  -- \
  pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1" \
    --use_prefetcher True \
    --repeat_batches 1 \
    --max_seq_len 1024 \
    --batch_size 1 \
    --max_generated_tokens 200 \
    --paged_attention True \
    --stop_at_eos 0 \
    --enable_trace \
    --mode full
```

Omitting `--num_layers` is intentional: both commands build and run every model layer.
