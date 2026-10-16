# r04-b02-a02 result: 1.6093 (ok). Regression: -1.1% vs the parent r04-b02-a01 (1.6275), just outside the ±1% noise band. The mechanism is real on kimi-k3 / mistral and costs the h7168 shapes

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline on every
  printed digit.
  - So the running-count act_valid, the pump's ack / send / receive order and the spin-pumped weight loop
    deliver exactly the same blocks into the same slots.
  - No hang, no JIT error. No kernel-config overflow: the reader grew +0.78 KB, the opt-in `RE_HWPROF`
    freed -1.46 KB of TRISC1 in profiler builds.
- I expected -5..-15% RE-DN everywhere, most on kimi-k3 / dsv4-flash / dsv4-pro / kimi-k2.7. Only kimi-k3 and
  mistral got it.

MATH zones, mean core per call over the 10 measured calls (/tmp/r04b04/zones2.py), parent -> this node:

| shape | chip µs | mean core end | GU | DN | crit core end (GU / DN) |
|---|---|---|---|---|---|
| dsv3 | 512.8 -> 520.1 (+1.4%) | 482 -> 484 | 327.6 -> 331.3 | 135.9 -> 133.6 | 512 -> 519 (354 -> 365 / 139 -> 135) |
| kimi-k2.7 | 717.3 -> 740.4 (**+3.2%**) | 708 -> 725 | 476.6 -> 485.3 | 207.5 -> 214.7 | 716 -> 739 |
| kimi-k3 | 1342.8 -> 1329.1 (**-1.0%**) | 1244 -> 1226 | 727.0 -> **776.9** | 492.6 -> **422.9 (-14%)** | 1342 -> 1328 |
| glm | 449.2 -> 451.7 (+0.5%) | 421 -> 425 | 284.1 -> 283.9 | 118.2 -> 122.3 | 448 -> 451 |
| mistral | 183.8 -> 178.7 (**-2.8%**) | 160 -> 157 | 106.0 -> 106.0 | 44.6 -> **41.0** | 183 -> 178 (crit DN 61 -> 46) |
| dsv4-flash | 308.7 -> 314.7 (+1.9%) | 287 -> 288 | 181.8 -> 184.4 | 95.5 -> 93.6 | 308 -> 314 |
| dsv4-pro | 1056.4 -> 1084.2 (**+2.6%**) | 1044 -> 1066 | 693.5 -> 693.4 | 337.0 -> **359.7 (+7%)** | 1056 -> 1083 |
| minimax-m3 | 372.9 -> 371.9 | 343 -> 339 | 228.6 -> 228.3 | 109.2 -> 106.7 | 372 -> 371 |
| gpt-oss | 255.8 -> 267.7 (+4.7%, crit-GU lottery) | 227 -> 229 | 151.1 -> 152.0 | 70.8 -> 72.0 | 255 -> 267 (GU 186 -> 201) |

## Why (best explanation)
1. **Where the activated stream was the chain, decoupling works.**
   - kimi-k3: DN -70 µs per mean core (-14%), the largest DN cut in the campaign. It has many small down blocks and
     a 57.5% activated wait in r04-b03-a01.
   - About 50 µs of it reappears as GU on the next expert. This is the DRAM-bound conservation seen in
     r02-b02-a03 / r03-b04-a03 / r04-b04-a01. The net is mean core end -18 µs, chip -1.0%.
   - mistral: DN -3.6 mean and crit DN 61 -> 46. The decoupling also shrank the tail there, chip -2.8% (its
     noisy shape).
   - minimax-m3 DN -2.5, dsv4-flash -1.9.
2. **On the h7168 shapes DN got slower: dsv4-pro +23 µs, kimi-k2.7 +7; GU +9 on k2.7.** No counters this run, so this is
   inferred from the code change and the shape pattern:
   - (a) **The act send is blocking and now lands inside the weight stream's critical waits.** pump_act's sender
     branch issues the act multicast and then waits `noc.async_write_barrier()` for every row ack (its own
     loopback copy). That is ~1-2 µs for a pcM x in0_block_w_d block.
     - Before, the in1 (column) sender always finished read barrier -> down_done -> weight multicast before
       touching the act stream. Now any spin (in1_ready, read barrier, down_done) can enter an act send, and the
       column's weight multicast waits behind it.
     - Likewise a receiver busy in its own act send acks in1 / pushes weights late.
     - h7168 down blocks are the biggest (per_core_N_d ≈ 21: ~72 KB bf4 per block). Every column receiver waits on
       that multicast.
   - (b) **Little run-ahead to gain on those shapes.** The act stream can lead only by the cb_in0_down_full
     ring depth. h7168 programs are the L1-tightest and most likely run d_slots = 2: a receiver acks a+1 only
     once its compute popped a-1. So they pay the cost of (a) for at most one block of slack. kimi-k3 / mistral
     plausibly have d_slots = 3.
     - Unverified; a child should log or derive d_slots per shape. The factory's `cb_footprint_bytes` with
       kDeepRingSlots is the place to check.
   - dsv3 / glm (also h7168 / h6144, i2048) moved +2..+4 DN: the same direction, smaller.
3. Removing r04-b02-a01's explicit down look-ahead lambda changed nothing by itself. The sender's next read is
   still issued as soon as its slot frees: the top-of-block reserve now directly follows the multicast.
4. gpt-oss +4.7% is the crit-core GU lottery (crit GU 186 -> 201, mean core +2 µs). Its DN is unchanged.

## Classification
repairable failure (bug-like execution issue, not a flawed idea). The decoupling is correct and bit-identical. On
the shapes with slack it gives the largest DN cut so far: kimi-k3 -14% DN, mistral tail cut. Its blocking act
send inside the weight-critical waits plausibly costs the h7168 shapes more than the decoupling gains there.

## What a child of this node should try next
1. **Make the act send non-blocking and keep it out of the weight-critical window** (same file, small change).
   - In pump_act's sender branch, drop `noc.async_write_barrier()`. Set a "sent" state and push the own block
     on a later pump call once `ncrisc_noc_nonposted_writes_flushed(noc.get_noc_id())` holds. Pop cb_activated
     there too.
   - On the in1 sender, skip the sender branch, i.e. pump only acks / receives, between
     `in1_ready` satisfied and the weight multicast issued.
   - Expected: keep kimi-k3 / mistral's DN gain and remove the h7168 DN growth. If dsv4-pro DN still grows, the
     cause is NoC contention between the earlier act multicasts and the weight traffic, not blocking.
2. **Check d_slots per shape.** Log the factory's choice, or recompute `cb_footprint_bytes`. If h7168 shapes run
   d_slots = 2, try a 3-slot cb_in0_down_full alone (activated only; it is small: pcM x in0_block_w_d bf16 tiles)
   so the act stream has real slack. Keep the down-weight ring at 2. That is a host change (factory + reader
   `pages_reservable` is already generic).
3. If neither helps, gate the decoupling per shape: e.g. only when `num_blocks_d * per_core_M` is small, or for
   the kimi-k3-like many-small-block case. Use the old lock-step loop elsewhere. That costs reader size.
4. `RE_HWPROF` is now opt-in. Define it at the top of fused_swiglu.cpp (and make room) when you need the MATH perf
   counters.
5. Don't retry: gate/up ack decoupling (r04-b04-a01), moving down-weight reads to the writer / NoC 1
   (r02-b04-a02).
