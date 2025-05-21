# Microsoft Phi 2

This is the basic compilation of Microsoft Phi 2 on Tenstorrent Wormhole, using the tt_transformers starter template & following the standard BERT implentation includes (mlp, attention, rotary embedding, sdpa)

## Dependencies
This implimentation does not require dependencies else which comes with the ttnn

## Run a demo
You expected to set the env variable WH_ARCH_YAMP as below:

```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

### 3. Run the demo

The `simple_text_demo.py` script includes the following main modes of operation and is parametrized to support other configurations.

- `batch-1`: Runs a small prompt (128 tokens) for a single user
- `batch-32`: Runs a small prompt (128 tokens) for a a batch of 32 users

```
# Examples of how to run the demo:

# Batch-1
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

# Batch-32
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"

```

The above examples are run in `ModelOptimizations.performance` mode. You can override this by setting the `optimizations` argument in the demo. To use instead the accuracy mode you can call the above tests with `-k "accuracy and ..."` instead of performance.

#### Custom optimizations
This is the basic, always you can customize the computing, refer to the documentation

### Expected performance and accuracy

See [PERF.md](PERF.md) for expected performance and accuracy across different configurations.
