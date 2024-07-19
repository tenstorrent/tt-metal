# lm-eval-harness.py

To run zero-shot evaluations of this model (for comparison against Table
3 of [Mamba: Linear-Time Sequence Modeling with Selective State Spaces
](https://arxiv.org/abs/2312.00752)) we use the `lm-evaluation-harness` library.

To install the `lm-evaluation-harness` library, run the following commands:

```sh
# Clone and then install as an editable package
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## CPU

Running against the CPU reference model (`MambaDecode`)

```sh
python benchmarks/lm_harness_eval.py --tasks hellaswag \
    --device cpu --batch_size 1 --limit 0.1 \
    --model mamba-cpu-reference
```
