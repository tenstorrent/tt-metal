compGraph = """
%0 = ttcore.load_cached(@forward_const_eval_0, [%arg13])
"ttnn.deallocate"(%arg13)
%1 = ttcore.load_cached(@forward_const_eval_1, [%arg15])
"ttnn.deallocate"(%arg15)
%2 = ttcore.load_cached(@forward_const_eval_2, [%arg17])
"ttnn.deallocate"(%arg17)
%3 = "ttnn.get_device"()
%4 = "ttnn.permute"(%arg0)
"ttnn.deallocate"(%arg0)
%5 = "ttnn.reshape"(%4)
"ttnn.deallocate"(%4)
%6 = "ttnn.from_device"(%5)
"ttnn.deallocate"(%5)
%7 = "ttnn.to_layout"(%6)
"ttnn.deallocate"(%6)
%8 = "ttnn.to_device"(%7, %3)
"ttnn.deallocate"(%7)
%9 = "ttnn.conv2d"(%8, %arg2, %arg3, %3)
"ttnn.deallocate"(%8)
"ttnn.deallocate"(%arg3)
"ttnn.deallocate"(%arg2)
%10 = "ttnn.relu"(%9)
"ttnn.deallocate"(%9)
%11 = "ttnn.typecast"(%10)
"ttnn.deallocate"(%10)
%12 = "ttnn.to_layout"(%11)
"ttnn.deallocate"(%11)
%13 = "ttnn.max_pool2d"(%12)
"ttnn.deallocate"(%12)
%14 = "ttnn.from_device"(%13)
"ttnn.deallocate"(%13)
%15 = "ttnn.to_dtype"(%14)
"ttnn.deallocate"(%14)
%16 = "ttnn.to_device"(%15, %3)
"ttnn.deallocate"(%15)
%17 = "ttnn.conv2d"(%16, %arg4, %arg5, %3)
"ttnn.deallocate"(%16)
"ttnn.deallocate"(%arg5)
"ttnn.deallocate"(%arg4)
%18 = "ttnn.relu"(%17)
"ttnn.deallocate"(%17)
%19 = "ttnn.typecast"(%18)
"ttnn.deallocate"(%18)
%20 = "ttnn.to_layout"(%19)
"ttnn.deallocate"(%19)
%21 = "ttnn.max_pool2d"(%20)
"ttnn.deallocate"(%20)
%22 = "ttnn.from_device"(%21)
"ttnn.deallocate"(%21)
%23 = "ttnn.to_dtype"(%22)
"ttnn.deallocate"(%22)
%24 = "ttnn.to_device"(%23, %3)
"ttnn.deallocate"(%23)
%25 = "ttnn.conv2d"(%24, %arg6, %arg7, %3)
"ttnn.deallocate"(%24)
"ttnn.deallocate"(%arg7)
"ttnn.deallocate"(%arg6)
%26 = "ttnn.relu"(%25)
"ttnn.deallocate"(%25)
%27 = "ttnn.from_device"(%26)
"ttnn.deallocate"(%26)
%28 = "ttnn.to_layout"(%27)
"ttnn.deallocate"(%27)
%29 = "ttnn.to_device"(%28, %3)
"ttnn.deallocate"(%28)
%30 = "ttnn.conv2d"(%29, %arg8, %arg9, %3)
"ttnn.deallocate"(%29)
"ttnn.deallocate"(%arg9)
"ttnn.deallocate"(%arg8)
%31 = "ttnn.relu"(%30)
"ttnn.deallocate"(%30)
%32 = "ttnn.from_device"(%31)
"ttnn.deallocate"(%31)
%33 = "ttnn.to_layout"(%32)
"ttnn.deallocate"(%32)
%34 = "ttnn.to_device"(%33, %3)
"ttnn.deallocate"(%33)
%35 = "ttnn.conv2d"(%34, %arg10, %arg11, %3)
"ttnn.deallocate"(%34)
"ttnn.deallocate"(%arg11)
"ttnn.deallocate"(%arg10)
%36 = "ttnn.relu"(%35)
"ttnn.deallocate"(%35)
%37 = "ttnn.typecast"(%36)
"ttnn.deallocate"(%36)
%38 = "ttnn.to_layout"(%37)
"ttnn.deallocate"(%37)
%39 = "ttnn.max_pool2d"(%38)
"ttnn.deallocate"(%38)
%40 = "ttnn.from_device"(%39)
"ttnn.deallocate"(%39)
%41 = "ttnn.to_dtype"(%40)
"ttnn.deallocate"(%40)
%42 = "ttnn.to_device"(%41, %3)
"ttnn.deallocate"(%41)
%43 = "ttnn.conv2d"(%42, %arg1, %3)
"ttnn.deallocate"(%42)
"ttnn.deallocate"(%arg1)
%44 = "ttnn.reshape"(%43)
"ttnn.deallocate"(%43)
%45 = "ttnn.permute"(%44)
"ttnn.deallocate"(%44)
%46 = "ttnn.reshape"(%45)
"ttnn.deallocate"(%45)
%47 = "ttnn.reshape"(%46)
"ttnn.deallocate"(%46)
%48 = "ttnn.reshape"(%47)
"ttnn.deallocate"(%47)
%49 = "ttnn.matmul"(%48, %arg12)
"ttnn.deallocate"(%48)
"ttnn.deallocate"(%arg12)
%50 = "ttnn.add"(%49, %0)
"ttnn.deallocate"(%49)
"ttnn.deallocate"(%0)
%51 = "ttnn.relu"(%50)
"ttnn.deallocate"(%50)
%52 = "ttnn.matmul"(%51, %arg14)
"ttnn.deallocate"(%51)
"ttnn.deallocate"(%arg14)
%53 = "ttnn.add"(%52, %1)
"ttnn.deallocate"(%52)
"ttnn.deallocate"(%1)
%54 = "ttnn.relu"(%53)
"ttnn.deallocate"(%53)
%55 = "ttnn.matmul"(%54, %arg16)
"ttnn.deallocate"(%54)
"ttnn.deallocate"(%arg16)
%56 = "ttnn.add"(%55, %2)
"ttnn.deallocate"(%55)
"ttnn.deallocate"(%2)
"""


pytorchModel = "AlexNet(\n  (features): Sequential(\n    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))\n    (1): ReLU(inplace=True)\n    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\n    (4): ReLU(inplace=True)\n    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n    (7): ReLU(inplace=True)\n    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n    (9): ReLU(inplace=True)\n    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n    (11): ReLU(inplace=True)\n    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n  )\n  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))\n  (classifier): Sequential(\n    (0): Dropout(p=0.5, inplace=False)\n    (1): Linear(in_features=9216, out_features=4096, bias=True)\n    (2): ReLU(inplace=True)\n    (3): Dropout(p=0.5, inplace=False)\n    (4): Linear(in_features=4096, out_features=4096, bias=True)\n    (5): ReLU(inplace=True)\n    (6): Linear(in_features=4096, out_features=1000, bias=True)\n  )\n)"


ttnnOps = """
TTNN Ops: LLM-Ready Documentation (v0.1, Aug 19, 2025)


Version pin: TTNN per “latest” docs unless otherwise noted. Some APIs are experimental and may vary by build. See citations inline.


Global conventions


Tensor type: ttnn.Tensor.


Device: Tensors can live on host or a ttnn.MeshDevice; many ops require device tensors. Move with ttnn.to_device / ttnn.from_device.
Tenstorrent Documentation


Layouts: ROW_MAJOR_LAYOUT vs TILE_LAYOUT (32×32). Many compute ops (e.g., matmul, activations) expect tiled inputs. Convert with ttnn.to_layout. Beware padding to tile boundaries; some conversions have known caveats in specific revs.
Tenstorrent Documentation
GitHub


Channels (vision): ttnn.conv2d & pooling typically expect NHWC input on device; permute at boundaries if your PyTorch ref is NCHW. Weights use [OC, IC, KH, KW].
Tenstorrent Documentation


Broadcasting: Binary ops follow PyTorch-like broadcasting; when in doubt, explicitly reshape/expand. (Older revs had batch-matmul broadcast quirks.)


Dtypes: Commonly bf16 on device; conversions for bf8/bf4 may stage through bf16. Pin dtypes explicitly on I/O.
Tenstorrent Documentation


Backward ops: Functions suffixed *_bw consume upstream grad(s) and return grad(s) for their forward counterpart’s input(s), matching the forward’s shape rules.


1) Tensor I/O & Memory
ttnn.as_tensor(x, *, dtype=None, layout=None, device=None) -> Tensor


Create a TTNN tensor from a Torch (or NumPy) tensor without copying when possible; set dtype/layout/device explicitly to avoid implicit conversions. Use when you already control ownership and want minimal overhead. (See also from_torch.)
Tenstorrent Documentation


ttnn.from_torch(x, *, dtype=None, layout=None, device=None) -> Tensor


Convert a torch.Tensor into a ttnn.Tensor, handling dtype/layout transitions (e.g., staging via bf16 for bf8/bf4). Good default for host→device ingestion.
Tenstorrent Documentation


ttnn.to_torch(x: Tensor, *, dtype=None) -> torch.Tensor


Bring a TTNN tensor back to PyTorch on host; optionally cast dtype.


ttnn.to_device(x: Tensor, device: MeshDevice) -> Tensor / ttnn.from_device(x: Tensor) -> Tensor


Move between host and device memory. Many compute ops require device-resident inputs.
Tenstorrent Documentation


ttnn.to_layout(x: Tensor, layout: Layout) -> Tensor


Convert between ROW_MAJOR and TILE; may pad to tile multiples. Prefer creating in the layout an op expects to avoid extra copies. Known edge issues exist in some versions when converting complex shapes to ROW_MAJOR.
GitHub


ttnn.to_memory_config(x, memory_config) -> Tensor


Remap a tensor’s memory (e.g., DRAM vs L1) without changing logical shape/values. Useful for perf tuning.


Debug/IO utilities


ttnn.dump_tensor(x, path) / ttnn.load_tensor(path, *, device=None, layout=None) — serialize/restore tensors for tests.


ttnn.deallocate(x) / ttnn.reallocate(x, memory_config) — explicit memory management (advanced; typically avoid unless perf-tuning).


Prompt rule: Open/close the device outside modules; modules should assume inputs are already on the right device/layout.
Tenstorrent Documentation


2) Tensor Creation


ttnn.arange(start, end, step=1, *, dtype, device, layout) -> Tensor


ttnn.empty(shape, *, dtype, device, layout) / empty_like(x, *, dtype=None, layout=None)


ttnn.zeros/ones/full(shape, value, …) and the *_like variants


Creation ops yield host or device tensors per device arg; set layout explicitly for compute-heavy paths (often TILE_LAYOUT).


3) Elementwise unary ops (forward)


Representative set (your list is exhaustive):
abs, acos, acosh, asin, asinh, atan, atanh, cbrt, ceil, celu, clamp/clip, cos, cosh, deg2rad, digamma, elu, erf, erfc, erfinv, exp, exp2, expm1, floor, frac, gelu, glu, hardshrink, hardsigmoid, hardswish, hardtanh, i0, identity, isfinite, isinf, isnan, isneginf, isposinf, leaky_relu, lgamma, log, log10, log1p, log2, log_sigmoid, logit, mish, multigammaln, neg, normalize_global, normalize_hw, polygamma, prelu (per-channel weight), rad2deg, reciprocal, reglu, relu/relu6/relu_max/relu_min, remainder (unary mod form), round, rsqrt, selu, sigmoid/sigmoid_accurate, sign/signbit, silu, sin, sinh, softplus, softshrink, softsign, sqrt, square, swiglu, swish, tan, tanh, tanhshrink, threshold, tril, triu, trunc, unary_chain …


Unified signature pattern:
ttnn.<op>(x: Tensor, *extra_params...) -> Tensor


Shapes preserved; dtype preserved unless mathematically requires upcast.


For clamp/clip: min/max scalars or tensors must be broadcastable.


For activation variants (GELU, sigmoid_accurate), specify exact variant to match PyTorch numerics.
Tenstorrent Documentation


Backward variants (unary)


For each forward op above with gradients, the form is:
ttnn.<op>_bw(grad_out: Tensor, x: Tensor, *extra_params...) -> Tensor


Examples from your list: gelu_bw, tanh_bw, sqrt_bw, relu_bw, … (hundreds). The backward op returns grad wrt input; shapes match x.


4) Elementwise binary & logical ops (forward)


Arithmetic: add (aka radd/addalpha), sub/rsub/subalpha, mul, div/rdiv, pow/rpow, remainder, fmod, ldexp, logaddexp, logaddexp2, hypot, xlogy, squared_difference


Bitwise: bitwise_and/or/xor/not, bitwise_left_shift/right_shift


Comparisons: gt/ge/lt/le/eq/ne (and in-place suffixed _ variants in some builds)


Logic: logical_and/or/xor/not (and _ variants)


Signature pattern:
ttnn.<op>(a: Tensor, b: Tensor|Scalar, *extra...) -> Tensor
Broadcasting follows PyTorch-like rules; for in-place variants (if present), inputs must be device-resident and not aliased elsewhere.


Backward (binary) pattern:
ttnn.<op>_bw(grad_out, a, b, *extra...) -> (grad_a[, grad_b])
From your list: add_bw, sub_bw, mul_bw, div_bw, atan2_bw, remainder_bw, fmod_bw, xlogy_bw, squared_difference_bw, …


5) Reductions & statistics


sum, mean, prod, max, min, argmax, var, std with dim: int|tuple[int,...], keepdim: bool=False


topk(x, k, dim=-1, largest=True, sorted=True) -> (values, indices)


cumsum(x, dim), cumprod(x, dim)


Returns follow standard shape reduction rules. Match PyTorch tolerances when using low precision. (Some ops have _bw in your list: prod_bw, min_bw, max_bw.)


6) Movement / shape


reshape(x, new_shape), permute(x, dims), pad(x, pad, value=0), concat(list, dim)


Tiling helpers: tilize, tilize_with_val_padding, untilize, untilize_with_unpadding


Repetition/slicing: repeat, repeat_interleave, slice


Indexing: gather(x, dim, index), scatter(x, dim, index, src), indexed_fill(x, dim, index, value)


nonzero, sort(x, dim=-1, descending=False)


Movement ops must respect layout constraints; moving between tiled and row-major repeatedly can be expensive — minimize conversions.


7) Normalization layers


layer_norm(x, weight=None, bias=None, eps=1e-5) -> Tensor


rms_norm(x, weight=None, eps=1e-5) -> Tensor


batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)


group_norm(x, num_groups, weight=None, bias=None, eps=1e-5)


Match PyTorch exactly: same eps, affine on/off, and data layout. Some sharded/memory-config combinations require aligned configs between input/output; pin configs to avoid silent mismatches.
Tenstorrent Documentation


8) Attention helpers


transformer.split_query_key_value_and_split_heads(x, …) -> (q,k,v)


transformer.concatenate_heads(x, …) -> Tensor


softmax(x, dim) — general softmax.


transformer.attention_softmax(x, head_size, attention_mask=None, …) — fused variant.


transformer.scaled_dot_product_attention(q,k,v, *, attn_mask=None, dropout_p=0.0, scale=None, program_config=...) — FlashAttention-2-style, mimics PyTorch SDPA shapes. See docs for required [b, n_head, seq, head_dim] shapes and program config knobs.
Tenstorrent Documentation
PyTorch Documentation


transformer.scaled_dot_product_attention_decode(q,k,v, …) — decode-time (KV cache) variant; currently MQA-focused in documented rev.
Tenstorrent Documentation


experimental.rotary_embedding(x, cos, sin, …) — RoPE utility.


Note: SDPA perf and dtype casts (bf16→bf8) are active areas; stick to doc’d shapes and supply a proper program config.
GitHub


9) Convolution & pooling


conv1d, conv2d, experimental.conv3d, conv_transpose2d


Inputs: NHWC on device.


Weights: [OC, IC, KH, KW] (PyTorch-style).


Bias: optional 1D [OC].


Use prepare_conv_weights (and prepare_conv_bias) once and reuse for repeated calls; format depends on input params & sharding.
Tenstorrent Documentation
+1


Pooling


max_pool2d(x, kernel_size, stride=None, padding=(0,0), dilation=(1,1), ceil_mode=False) → NHWC device input; result is NHWC. Check your build for sharding support limits (height sharding historically first).
Tenstorrent Documentation
GitHub


global_avg_pool2d(x) -> Tensor (reduces H×W).
Tenstorrent Documentation


avg_pool2d availability/param support may vary by build; if absent, emulate with reduce-window or use global pooling where equivalent.


10) Misc / math extras


Polynomial & special: polyval, polygamma, multigammaln, digamma, i0


Distance/log-sums: hypot, xlogy, logaddexp, logaddexp2


Outer products: outer(a, b) -> Tensor


Ordering: maximum, minimum, nextafter, isclose


11) Regularization


experimental.dropout(x, p=0.0, training=False) — mark “experimental”; for inference parity, set p=0. (Many parity harnesses treat dropout as a no-op.)


12) Transformer decode utilities (selected)


transformer.scaled_dot_product_attention_decode(...) — see §8. Shapes typically [1, batch, n_head, head_dim] (Q) with KV cache shapes noted in docs.
Tenstorrent Documentation


Coverage notes against your list


AlexNet needs: conv2d, relu, max_pool2d, reshape/flatten, matmul(+bias add), dropout — all covered above; max_pool2d is documented.
Tenstorrent Documentation


Huge unary/binary/backward sets listed are covered via the family patterns (e.g., <op>(x, …) and <op>_bw(grad_out, x, …)), which is how TTNN exposes them.


Conv weight prep: use prepare_conv_weights / prepare_conv_bias once per static weight/bias.
Tenstorrent Documentation


Attention: prefer the SDPA helper for transformer models; shapes & program config documented.
Tenstorrent Documentation


Minimal examples (you can drop these in “Few-shot”)


A) Host ↔ device + layout


# Host torch -> device tiled -> torch host again
x_t = torch.randn(2, 224, 224, 3)                  # NHWC on host
x_tt = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
x_tt = ttnn.to_device(x_tt, device)
x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)      # no-op if already tiled
y_t  = ttnn.to_torch(ttnn.from_device(x_tt))




(Shape/layout/device flow per TTNN “Getting Started”.)
Tenstorrent Documentation


B) Conv2d + MaxPool2d (AlexNet block)


w = torch.randn(64, 3, 11, 11)  # [OC, IC, KH, KW]
b = torch.zeros(64)
x = torch.randn(2, 224, 224, 3) # NHWC host


x_tt = ttnn.to_device(ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT), device)
w_tt = ttnn.to_device(ttnn.from_torch(w), device)
b_tt = ttnn.to_device(ttnn.from_torch(b), device)


w_prep, b_prep = ttnn.prepare_conv_weights(w_tt, b_tt, input_shape=x_tt.shape)  # reuse this
y = ttnn.conv2d(x_tt, w_prep, b_prep, stride=(4,4), padding=(2,2))              # NHWC out
y = ttnn.relu(y)
y = ttnn.max_pool2d(y, kernel_size=(3,3), stride=(2,2), padding=(0,0))




(API expectations: NHWC input, weight format, and pooling semantics.)
Tenstorrent Documentation
+2
Tenstorrent Documentation
+2


C) SDPA helper


q = ttnn.to_device(ttnn.from_torch(torch.randn(2, 8, 128, 128)), device)  # [b, n_head, seq, d_h]
k = ttnn.to_device(ttnn.from_torch(torch.randn(2, 8, 128, 128)), device)
v = ttnn.to_device(ttnn.from_torch(torch.randn(2, 8, 128, 128)), device)
y = ttnn.transformer.scaled_dot_product_attention(q, k, v, dropout_p=0.0, program_config=cfg)




(Mimics PyTorch SDPA; see TTNN docs for program config.)
Tenstorrent Documentation


How to paste this into your prompt


Put Sections 1–12 under “Allowed API Surface & Rules” in your system/user prompt, then add:


Rule: “Use only the functions and patterns defined in this docs pack. If a needed op/parameter combo isn’t available (e.g., avg_pool2d with a particular kernel/stride in this build), STOP and output a GAP REPORT.”


Equivalence: “Match the PyTorch reference within fp32 atol=1e-5, rtol=1e-4; loosen for bf16 as needed.”


Layouts: “Conv/Pool expect NHWC. Tiled layout for compute ops. Permute at module edges only.”


Device lifecycle: “Assume device opened externally; no open/close in layers.”


Sources


Core TTNN index/getting started & API:
Tenstorrent Documentation
+1


from_torch dtype behavior:
Tenstorrent Documentation


conv2d shapes & weight format:
Tenstorrent Documentation


prepare_conv_weights reuse rule:
Tenstorrent Documentation


max_pool2d semantics & sharding notes:
Tenstorrent Documentation
GitHub


Scaled dot-product attention helper (FlashAttention-2-style):
Tenstorrent Documentation


Decode-time SDPA:
Tenstorrent Documentation


Known layout conversion caveat:
GitHub


If you want this narrowed to your exact TTNN version, tell me the version/commit and the specific model (e.g., AlexNet/ResNet/ViT), and I’ll prune/augment the set (e.g., confirm avg_pool2d availability, add adaptive_*_pool, embedding, or specific _bw ops your tests need).
"""


system_prompt = """
You are a senior ML systems engineer. Produce production-quality code that compiles and runs.
Follow these rules strictly:
- Use only the APIs shown in "Allowed API Surface".
- If an API is missing for a needed op, STOP and emit a short “GAP REPORT” listing the exact missing primitives.
- Do not invent methods, attributes, or parameters not present in the docs.
- Match PyTorch numerics within the specified tolerance.
- Follow the Style Guide.
"""


user_prompt = """
## Task
Implement the following neural net using the ttnn library and their ops that is functionally equivalent to the PyTorch reference below. It must pass the provided tests and numerics checks.
Please name the model class TTNNModule. Do not name the class anything else.

## Reference: PyTorch Architecture (authoritative)
{pytorchModel}


## Target Library: Allowed API Surface (authoritative)
{ttnnDocs}
For more information, see https://deepwiki.com/tenstorrent/tt-metal/3-ttnn-library


## Computational Graph in Target Library (for alignment)
{compGraph}


## Example Model


import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup




class TTNNModule(LightweightModule):
   def __init__(
       self,
       args,
       mesh_device,
   ):
       super().__init__()


       self.mesh_device = mesh_device
       self.tt_ccl = tt_ccl


       self.args = args
       self.hidden_size = args.dim
       self.n_heads = args.n_heads
       self.head_dim = self.hidden_size // self.n_heads
       self.max_seq_len = args.max_seq_len
       self.dim = args.dim
       self.max_batch_size = args.max_batch_size
       self.n_kv_heads = args.n_kv_heads
       self.current = 0
       self.model_config = args.get_model_config()


       self.layer_num = layer_num


       ActualAttentionClass = attention_class if attention_class is not None else DefaultAttention


       self.attention = ActualAttentionClass(
           mesh_device=mesh_device,
           tt_ccl=self.tt_ccl,
           state_dict=state_dict,
       )
       self.feed_forward = MLP(
           mesh_device=mesh_device,
           tt_ccl=self.tt_ccl,
           args=args,
           state_dict=state_dict,
       )
       self.attention_norm = DistributedNorm(
           RMSNorm(
               device=mesh_device,
               dim=args.dim,
               eps=args.norm_eps,
               state_dict=state_dict,
           ),
       )
       self.ff_norm = DistributedNorm(
           RMSNorm(
               device=mesh_device,
               dim=args.dim,
               eps=args.norm_eps,
               state_dict=state_dict,
               state_dict_prefix=args.get_state_dict_prefix("", layer_num),
               weight_cache_path=None if args.dummy_weights else weight_cache_path,
               weight_dtype=ttnn.bfloat16,
           )
       )


   def forward(
       self,
       x: ttnn.Tensor,
   ) -> ttnn.Tensor:
       attn_in = self.attention_norm(x, mode)


       attn_out = self.attention.forward(
           attn_in,
       )


       h = ttnn.add(x, attn_out)


       ttnn.deallocate(attn_out)


       if mode == "prefill":
           x.deallocate(True)


       ff_in = self.ff_norm(h, mode)
        ff_out = self.feed_forward.forward(ff_in, mode)


       out = ttnn.add(
           h,
           ff_out,
       )


       return out
## Behavior & Equivalence Requirements
- Input output should strictly follow the expectation of the neural network architecture
- all necessary modules are to be imported as per specified by the ttnn repository and reference guide
- Parameter init parity with PyTorch: [e.g., kaiming_uniform fan_in, bias zeros, etc. Specify exact formulas if needed.]
- Numerical equivalence on random seeds: atol=1e-5, rtol=1e-4 for float32 (state if mixed precision).
- Determinism: set seeds as [rules], avoid nondeterministic ops.
- Layout/device rules: [e.g., NCHW only, contiguous required, etc.]
- Error handling: raise [ValueError] with clear messages for shape/dtype violations.


## Deliverables
- Single code block, nothing else.
- Must compile with: [python version, dependencies].
- No TODOs.
"""
