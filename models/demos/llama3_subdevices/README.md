# Llama3 TG

This codebase includes Llama3.1-70B on TG.

### Prefill ccl ops
When running `text_demo.py` on a machine with torus, all ops will by default use ring topology. To use line implementation of ops you can set enviroment variables:
- LINE_RS = 1: to use line for all ReduceScatter ops
- LINE_AG = 1: use line for all AllGather ops

To use line for only some of the AG ops, you can set USE_LINE_AG set in `llama_ccl.py`, for example to use line for all RS and just QKV AG, and ring for the rest of AG set:
- LINE_RS = 1
- LINE_AG = 0
- USE_LINE_AG = {"QKV"}
