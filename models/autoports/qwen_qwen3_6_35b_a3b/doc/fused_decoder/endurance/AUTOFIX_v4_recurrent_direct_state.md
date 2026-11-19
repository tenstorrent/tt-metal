# Diagnostic: recurrent add writes the stable DRAM state

Base is immutable `selected_fused_decoder_v4.py.snapshot`, SHA256 `b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c`. Candidate `v4_recurrent_direct_state.py.snapshot` is SHA256 `6bd85fe6b3232f8ced7babf72c49b083ebaf6852441b9bf099e42221cf1e2b1a`. Only `_recurrent_step` and its decode/tail caller change. No production, shared native, core harness, README or earlier probe was edited. No device was run by this investigation.

`AUTOFIX_v4_remaining_decode_audit.md` correctly identifies that final-copy removal changes the Q-contraction input placement. The earlier recurrent-boundary fusion folded BF16 cache rounding into decay, but retained the final copy; it is not a test of this producer/destination rewrite. A snapshot/patch search found no prior direct recurrent output-state writer control. Native FDR, FMA and altered reduction-order failures are not reopened here.

## Complete graph change

`_recurrent_step` gains optional `output_state=None`. The existing separate FP32 add becomes:

```python
new = ttnn.add(decayed, reshape_of_update, output_tensor=output_state)
```

Decode and internal singleton-tail callers pass the persistent `state["recurrent"]`. Default private callers remain functional and allocate a new result as before. The Q multiply explicitly requests L1 output, maintaining the original Q-product/contraction residency despite `new` now being DRAM. There is no intermediate DRAM-to-L1 state copy. Prefill's sequential state-correction loop and its final copy remain unchanged; only decode/tail suppresses the now-redundant final copy.

The stable state remains FP32 TILE DRAM `[1,32,128,128]`, with the same tensor object and buffer address. Public BF16 cache rounding remains at the original decay input activation, controlled by `cache_boundary`; internal tail steps still use their original boundary flag. No additional rounding, FMA, reduction regrouping, or query-scale change is introduced.

## Native output and alias safety

This candidate does **not** ask the add to overwrite either of its own operands. `decayed` and `update` are independent L1 temporaries. The destination is the older persistent DRAM state, consumed by the earlier decay multiply. Subsequent memory/delta computations consume `decayed`, not the old persistent state. In the existing single-command-queue graph, all those reads precede the add's store; the Q contraction then consumes the new state. No concurrent-call-on-one-state guarantee is introduced.

BinaryNG validation accepts interleaved TILE inputs and a preallocated output with the broadcasted logical shape. It checks any explicit output dtype against the destination dtype (`binary_ng_device_operation.cpp:381–385`), validates output tile geometry (`:480–501`), and uses the destination TensorSpec after the shape check (`:525–549`). `create_output_tensors` returns that destination directly (`:597–601`); its memory configuration takes precedence for the writer (`:693–697`). Thus passing the stable DRAM tensor is a real destination binding, not an ignored memory hint.

The two operands and destination are FP32. Equal-FP32 ADD selects SFPU (`binary_ng_device_operation.cpp:49–58`), and the factory uses direct FP32 unpack and FP32 destination (`binary_ng_program_factory.cpp:1370–1400`). No explicit accurate-mode flag is added, since explicit false is invalid for FP32 ADD under this binding. The unchanged dtype/shape arithmetic follows the same add implementation with a different output address/residency.

Native cached-program rebinding is slot-based, not inferred from buffer identity: `binary_ng_program_factory.cpp:1560–1580` rebuilds every runtime tensor binding on cache hits, and `:1642–1655` maps CB0/1/2 to the current inputA/inputB/output buffers. Its comments explicitly cover in-place aliases, though this candidate's add operands do not alias its destination. Reader/compute/writer CB synchronization remains native. The diagnostic must still establish stable addresses and independent-request rebinding on the installed runtime.

## Cost tradeoff

The2MiB updated state was previously written to L1, read there by Q contraction, then copied from L1 to DRAM. The candidate writes the add directly to DRAM and Q contraction reads it from DRAM. It eliminates one copy dispatch and the intermediate L1 updated-state allocation/write, but introduces a DRAM read for Q contraction. DRAM final-state writes are still required and are not counted as eliminated. Keeping Q products in L1 avoids accidentally moving another2MiB product plus reduction chain to DRAM. No performance gain follows from removing a node; the complete recurrence must be measured.

## Bounded exact and trace discriminator

`tests/v4_recurrent_direct_state_probe.py` extracts the actual decode caller and final-store boundary from each snapshot, then invokes the actual `_recurrent_step` and `_recurrent_sum`. It uses saved real Q/K/V/g/beta and initial-state captures at positions128/130/7703. These are historical numerical fixtures, not a new HF acceptance oracle. V's captured BF16 dtype is retained. The optional explicit decay fixture is computed once on CPU from captured g and supplied identically to both variants; the original g-to-SFPU-EXP route is tested separately.

Checks include:

- Complete core output and persistent state for each fixture with round-state on/off and explicit-decay/EXP paths.
- Twelve sequential updates mixing public-rounding and internal no-rounding flags, state snapshots after a four-step prefix, clone/copy restoration and exact suffix replay.
- Stable persistent tensor identities, addresses, FP32 dtype and DRAM placement; the candidate's returned new-state buffer must equal its destination.
- A second independent request state after cache reuse, with exact output/state and proof the first request's saved state is untouched.
- Warmed traced complete recurrence timing, including Q contraction and either final copy or direct add writer. Resets/readbacks are outside measured steps. Both variants use identical replay counts; first/final core and state are compared, then reset and replayed for exact repeatability.
- Captured twelve-step mixed-boundary continuation chains, replayed twice after reset; every retained output and final state is compared across repeats and variants.

This is recurrence-boundary coverage, not execution of the unchanged native PF chunk path or full convolution/decoder. Parent whole-decoder/endurance tests remain necessary before selection.

`v4_recurrent_direct_state_cpu.json` verifies finite fixture shapes, extracts the actual boundary statements, and proves the recurrent arithmetic AST is identical after removing only the new output keyword and explicit Q-product placement. `_recurrent_sum` is AST-identical. This CPU setup check makes no device numerical claim. Snapshot/probe/capture/CPU hashes are recorded. The device branch uses physicalchip0,32MiB trace reservation, L1-small32768, clean timing environment checks, synchronized trace release and `finally` close.

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_recurrent_direct_state_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_recurrent_direct_state_probe.json
```

Source legality is supported, but exact installed-device results, trace address behavior and complete recurrence latency remain pending. Neither source inspection nor a hypothetical copy saving warrants selection.

## Parent-owned device discriminator passed

The parent subsequently completed `v4_recurrent_direct_state_probe.json`, SHA256 `f0eceea5140e39e5c6ebf56d6090f2f034863037e84fb7a775a4c12bf885e911`. It reports `status=exact`,117 exact comparisons and clean device close. Its candidate and probe hashes match the frozen artifacts. The provenance sidecar binds the report and timing results.

Actual passing coverage includes both rounding modes and explicit-decay/EXP paths on all three real fixtures; core and persistent-state outputs; twelve-step mixed-boundary chains; prefix-state restoration and suffix repeatability; an independent second request with the first state unchanged; first/final timed trace results; reset/replay equivalence; and twice-replayed captured continuation chains. Persistent addresses remained6178432 and6440576 for the two independent variants, and the destination-object/dtype/DRAM assertions passed. These addresses are diagnostic observations from that session, not portable constants.

Complete recurrence traced medians were0.171889761ms for baseline and0.168955140ms for the direct writer: approximately2.935µs (1.71%) lower. Both timings include Q contraction and their respective persistent-state store. Reset/readback are outside measured steps, as documented above. This is a modest isolated recurrence result; whole-decoder timing and integrated acceptance remain pending. No production selection is claimed.

Separately, the parent reported a profiler abort preserved at `profile_v4_failed_01` and launched independent AutoDebug. That instrumented run is not used as evidence for this candidate. The parent reports uninstrumented correctness/timing and a subsequent trivial-add health smoke passed without reset. This memo makes no diagnosis of the independent profiler issue and does not use it to waive any validation gate. No hardware, source or harness changes were made by this documentation update.

## Parent whole-decoder result and isolated combined writers setup

The source-bound S256 decoder benchmark `v4_recurrent_direct_state_s256.json` reports107.707752ms prefill and1.313872868ms traced decode. Its SHA256 is `879db882a9594c5a35f6601f8986e682966de14ad484a0a5dc67ac5fe49d7878`. The contemporaneous b163 baseline `v4_persistent_rm_cache_baseline_s256.json` reports1.317830756ms decode, so this isolated writer improves that whole path by about3.958µs (0.30%). Prefill PCC0.999962395060449 and eager/traced decode PCC0.9999779064754709 are unchanged; execution is deterministic. These diagnostic short-run results do not establish full acceptance.

The parent's separately measured RM-cache controls report1.300121658ms decode for RM storage plus copy, and1.299049472ms for RM storage plus direct slice writer, with the same PCC values. These component medians are not added to predict combined performance.

Prepared isolated `v5_cache_writers.py.snapshot`, SHA256 `9d33bc14b5f349a9dabfbd4e6ce6344c31047fbcfcc2716f35de0133e2d8b7f7`, combines exactly:

- RM cache/direct-slice snapshot `v4_persistent_rm_cache_slice.py.snapshot`, SHA256 `ef2b0d13cd1217144ccc7de087d4b43631144ffee6a99f445577e46e63c9251e`: allocate BF16 conv cache in RM, consume it directly in RM concat, and write carry through the slice output tensor.
- This recurrent writer snapshot, SHA256 `6bd85fe6b3232f8ced7babf72c49b083ebaf6852441b9bf099e42221cf1e2b1a`: optional direct recurrent-state add destination, preserved L1 Q-product residency, and unchanged PF correction/copy.

Base remains b163. The combined patch and `.py.provenance.json` bind every source and component benchmark hash. Source proof applies the exact three RM substitutions and five recurrent substitutions in both orders: both produce byte-identical combined source. Undoing either component restores the other frozen snapshot byte-for-byte. Changed methods are only `new_recurrent_state`, `_recurrent_step`, and `_linear_attention`; their base/combined AST hashes are recorded. `norm` and every other method are AST-unchanged. There is no native-head norm candidate or other normalization change in this composition.

Conv and recurrent writers target different stable buffers at their original dependency points. Component legality and individual exact results support testing the composition, but do not prove its joint allocation/trace/performance behavior. No combined device execution or selection is claimed. Production, core harnesses, old snapshots and probes remain untouched; root owns the combined benchmark/endurance queue.
