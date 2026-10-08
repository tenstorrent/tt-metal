#!/bin/bash
# t283 attempt 5: 8-bit = LTX_QUANT=all_bf8_lofi with bf16 activations (LTX_QUANT_ACTIVATIONS=0). The default bf8
# activation cast feeds a BFLOAT8_B tensor into dit_fused_distributed_rmsnorm, which only takes bf16/fp32 (job 078).
# Gate fold off as in bf16nf. Everything else is run283d.sh unchanged. Usage: run283e.sh bf8wnf
export LTX_QUANT_ACTIVATIONS=0
exec bash /var/tmp/fasth3/t283/run283d.sh "${1:-bf8wnf}"
