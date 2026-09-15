# Scalar padded P16 alias probe

The separate-P16 experiment needed 1,474,560 CB bytes/core. The observed
111,616 reserved bytes made it exceed Blackhole's 1,572,864-byte L1 by 13,312.
This variant retains Q256/K512/D128 and every input slot, but aliases the
probability view onto score storage with an identical page pitch:

| View | Format | Pages | Page pitch | Shared allocation |
| --- | --- | ---: | ---: | ---: |
| CB6, QK scores | FP32 | 128 | 4096 B | 524,288 B |
| CB7, probabilities | BF16 | 128 | 4096 B | same allocation |

Total CB payload is **1,212,416 B/core** for both BF16-P and FP32 controls.
Adding the observed reservation leaves an estimated **248,832 B** free; actual
device allocation remains authoritative.

## Address and lifetime audit

Scalar `pack_tile<true>` computes each tile address from the CB's configured
`fifo_page_size` (`llk_pack_common_api.h:74–82`), not its native datatype size.
Thus BF16 P tile i writes 2048 bytes starting at `base + 4096*i`, after the
corresponding FP32 score tile has been read for max/sub/exp. Unlike compact
2048-byte pitch, it cannot overwrite score tile i-1 or i+1. The unused second
half of the page can remain stale: BF16 unpack does not consume it.

The matmul unpack API reads its input tile sizes from `fifo_page_size`
(`llk_unpack_AB_matmul_api.h:145–149`), so both PV and the P-times-ones
denominator traverse the same padded BF16 pages. Padded standalone roundtrip
tests support this addressing behavior, but do not qualify attention's lifetime
and synchronization by themselves.

**Physical aliasing does not share FIFO counters.** The existing P16 protocol
is intentionally retained, rather than reverting to CB6-only publication:

1. Reserve 128 pages in CB6 and independently in CB7 before each K chunk.
2. Publish score rows on CB6 as before. Publish each completed P row on CB7
   after all its column subblocks are packed; PV waits on CB7.
3. Keep both write pointers at the allocation base during the chunk.
4. Complete the last-row exp drain and existing PACK→UNPACK handshake.
5. After P consumers finish, pop 128 pages from each queue. Equal capacities
   and page pitches wrap both read pointers back to the shared base.

Reserving both queues also prevents reuse until both previous-chunk lifetimes
have ended. Publishing only CB6 would not satisfy CB7 consumers and would risk
deadlock; treating score readiness as P readiness would instead expose stale
or incompletely transformed values.

## Packing modes and replay ownership

Scalar remains the default. `--p-pack-width 4` is an explicit opt-in: FP32 P
uses ordinary width4; BF16 P uses the private `padded_pack.hpp` MOP, not native
contiguous BF16 blocked packing. The standalone `padded_pack_probe` passed all
128 tiles exactly on Blackhole with both DST modes, as reported by the parent.
This attention integration still needs its own qualification.

The custom MOP emits a Last per tile and sets output-channel Y stride to 64
bytes, giving BF16 tile starts 4096 bytes apart. There is one C++ pack call but
still four tile-close events. Output stride is restored to zero after every
P batch. The cached pack width is then invalidated so the ordinary width4 QK
MOP is necessarily restored even though the numeric width did not change.

Blackhole `ckernel_structs.h` declares only 32 replay words per thread. Slots
32–46 are NOT usable. The private pack program is reloaded into 17–31 after
every completed exp batch (after WAIT_SFPU). Native exp executes slots 0–7,
which remain intact. Cubic exp reloads its refiner into 8–21 on its next call,
then this pack helper reloads 17–31 again. Separate threads have separate
replay buffers, so QK/PV math replay is not touched. Later online-correction
code may overwrite pack replay; the next exp-phase init resets native exp.

This costs a 15-instruction replay upload plus MOP/stride configuration per
four P tiles. It may be slower despite reducing C++ pack call overhead; do not
claim a speedup before measuring. No CB sizes, FIFO protocol, input buffering,
chunk sizes, or numerical arithmetic were changed by this opt-in.

## Numeric and execution controls

Both drivers accept `--p-format fp32|bf16` and `--native-exp`; default exp is
the unbiased cheap cubic fit. Both use LoFi QK/PV, FP32 DST/recurrent state and
a denominator matched to the effective stored P. Q is RNE7/BF16; K/V are
RNE5/native-BFP8. No P SFPU pre-rounding is added. FP32 control retains its
original in-place CB6 protocol; its same-format CB7 alias view is unused.

Suggested initial distinct-KV, all-Q smoke (fresh label for every run):

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/padded_p16_streaming.py \
  --label padded_p16_bf16_cubic_4k_smoke --p-format bf16 \
  --length 4096 --heads 2 --cores 4 --sample-rows 4096 \
  --check-preprocess --iters 0
```

Repeat for FP32 and native exp before timing. Fullchip defaults remain
N8192/H10/110 cores with per-head KV chains; resident defaults repeat one
resident K/V block with no recurring input DM. Only the latter's final Q block
is saved, so resident correctness alone cannot test changing maxima.

Source hashes are captured before execution and checked again afterward,
including the private header, reused readers/writers/preprocessing, exp helper,
and LLK page-address helpers. New files only; existing P16/P8/active/frozen files
remain unchanged. Device allocation, JIT, attention correctness and timing have
not been run by this implementation agent.
