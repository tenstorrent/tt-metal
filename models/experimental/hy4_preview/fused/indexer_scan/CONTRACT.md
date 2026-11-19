<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Index-key scan: what a chip exposes to the cross-chip top-k merge, and what it takes back

Piece 4 (`fused/indexer_scan`) is chip-local. Piece 6 (`fused/merges`) turns the 32 chips' local top-2048 lists
into the global top-2048 of each user; only scores travel, every chip keeps its own winners. This file is the
interface between the two. Everything below is per chip and per user.

## 1. What the chip holds after the scan (`Scan`, mode "tree")

A **survivor**: 2048 32-bit words on the user's root scan core (`scan_core(user, 0)`), also written to DRAM
(`Scan.out`, uint32 tiles, pages `user * 16` and `user * 16 + 1`).

    word = [ BF16 score (bits 31:16) | stamp (bits 15:0) ]

* **Order.** The words are sorted as FP32 numbers, best first: by BF16 score, and equal scores by stamp (for a
  positive score the larger stamp is first, for a negative one the smaller). The entry of rank `q` (0 = best) is
  word `((q & 127) << 4) | (q >> 7)` of the two tiles laid end to end (`indexer_scan.rank_word`): the sorted
  sequence runs down the 16 columns of the [128 x 16] arrangement of the tiles' face rows. There is no
  rank-contiguous copy in memory; a consumer that wants one copies 16-word runs after a face transpose, or indexes
  by the formula (the cut kernel does the latter).
* **Stamp.** `[chunk 15:11 | block in tile 10:6 | tile in chunk 5 | position in block 4:0]`, with
  `chunk = bank * chunks_per_bank + (score tile of the bank) / 2`. Stripe row (the row of the chip's stripe of the
  user's slot, the id the sparse row reader takes):

      bank_block = ((chunk % chunks_per_bank) * 2 + tile) * 32 + block
      row        = ((bank_block * 8) + chunk / chunks_per_bank) * 32 + position

  (`Scan.stamp_row`, `kernels/cut.cpp`). 32 chunk ids bound a stripe to 65,536 rows per user per chip.
* **Entries the query does not see.** A chip always holds 2048 entries. Rows after the user's context score
  NEG = -(2^100) (BF16 0xF180) and padding lanes -inf (0xFF80); both sort last. An entry is real iff its score is
  above NEG / 2. A chip that sees fewer than 2048 rows of the user has fewer real entries.
* **Scores are BF16.** Two rows whose FP32 scores round to the same BF16 value are tied and ordered by stamp. The
  global order the merge builds must restrict, on every chip, to that chip's rank order: order equal scores by
  (anything per chip, then the chip's rank), never by row id directly.

## 2. What the merge needs from a chip, and the two ways to give it

**(a) The sorted score list.** The 2048 BF16 scores in rank order (4 KB per user per chip; the upper halves of the
survivor words). A merge that exchanges these (tagged with the chip) can count, for every chip, how many of the
global top-2048 are its own: `keep`. The first `keep` entries of the chip's rank order are its winners. Not built
here: the rank-ordered copy for the fabric sender (see "Order" above for where the words are).

**(b) Threshold queries.** "How many of your scores are >= t" for a BF16 value `t`: the cut program below with
`ties = 0x7fff` answers it (row 0 of its table, per user), in one program launch for all 8 users. A merge by
bisection on `t` costs one cross-chip round per step, so (a) is the better default; (b) is there for a final
tie-resolving round or a merge that works on sampled splitters.

## 3. The cut: global decision in, the chip's own winners out (`Cut`, `kernels/cut.cpp`)

Input: one uint32 per user (`Cut.cuts`, a 64-byte DRAM page, word `u` for user `u`; rewritten in place per step):

    bit 31 = 1   bits 15:0 = keep                  the chip keeps its `keep` best entries
    bit 31 = 0   bits 15:0 = t (BF16), 30:16 = ties
                 keep = #(score > t) + min(ties, #(score == t));  ties = 0x7fff: all of them

`keep` is clamped to the real entries. Output: the table the sparse row reader of piece 3 takes
(`fused/sparse_mla/layout.py selection_table`): uint32 `[1, 1, 1 + users, 2048]` ROW_MAJOR in DRAM,

    row 0, word u     count of user u
    row 1 + u         user u's stripe rows, best first; only the first count entries are written

One data-movement kernel on each user's root scan core: two binary searches over the rank order, then `keep`
stamp decodes. Counts reach user 0's core through one program semaphore per user and that core writes row 0.

## 4. Inputs of the scan (what the query producer has to provide)

* `Scan.qt` `[users, 1, 128, 32]` BF16 tiles in DRAM: the user's 32 query heads as COLUMNS, in the cache's frame
  (`tt/frames.py`: `[rope 64 interleaved | pass 64]`, Hadamard basis), with the head weight's magnitude folded in:
  `qt[u][:, h] = |w[u, h]| * q[u, h]` (`indexer_scan.device_query`).
* `Scan.step` uint32 `[1, 1, 8, 32]` ROW_MAJOR in DRAM: one 128-byte page per user, rewritten in place per step
  (`kernels/scan_common.hpp step_*`):

      word 4        positions the user sees (its query position + 1; the query's own key is in the cache
                    before the scan). The reader derives the chip's and the bank's visible prefix from it with the
                    block-cyclic arithmetic (128 tokens per chip per turn, 32 chips).
      word 5        BF16 bits of the user's score centre c (0: none). Scores then leave as BF16(score - c), in the
                    survivor, in the sorted list and in the cut's threshold alike. Every chip must be given the
                    same c for a user; the user's global cut of the previous step is the natural value.
      words 16..31  the 32 head-sum multipliers as BF16: sign(w[u, h]) = +-1 when the magnitude is folded into
                    the query, else w[u, h] itself (`device_query(fold=False)`, at the multiplier's precision).

* The head-sum selector tiles (32 per scan core: tile `c` holds the multipliers in column `c`, row = head, zeros
  elsewhere) are either read from a DRAM tensor the host fills (`ScanConfig.selectors = "dram"`, `Scan.sel`
  `[users, 32, 32, 32]` BF16) or live in an L1 tensor behind the selector CB, where the reader rewrites just the
  32 x 32 multipliers from the step page every run (`"l1"`).
* The key cache itself: the prefill stack's tensor, read in place; `slots[u] = user * 21 + full_rank(layer)`.

## 5. As a fragment of a layer program

`Scan.append_to(desc, coord, semaphore_base)` appends the piece to another program's descriptor for one chip:
3 kernels on the 64 scan cores (which must carry nothing else), 11 circular buffers on them, 2 program
semaphores. `Scan.io_tensors()` are the tensors to pass to `generic_op` along with the caller's. `Cut` is a
separate small program today (it reads the survivor from DRAM); inside a layer program its kernel would run on the
root scan cores after the merge and read the survivor from the root's own L1.
