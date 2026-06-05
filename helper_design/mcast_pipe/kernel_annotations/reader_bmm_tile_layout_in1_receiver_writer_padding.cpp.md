# reader_bmm_tile_layout_in1_receiver_writer_padding.cpp — annotation

Role: **RECEIVER half** (reader portion) + unrelated WRITER portion. Object API. Pairs with in1_sender_writer_padding.

## Fork signature (receiver view)
- **F1**: writer portion uses `noc.async_write_barrier()` (out drain, L764-equivalent) — not part of the Pipe block. Receiver itself issues no flush.
- **F2**: LEVEL FLAG, exact. `receiver_sem.wait(VALID)` (L133, L151). No wait_min here.
- **F3 / pre_handshake**: receiver back-half — `sender_sem.up` before `wait(VALID)`.

## Protocol steps
- L105-109: ctors; `sender_sem` cta4, `receiver_sem` cta5.
- in1 block (L122-136): `cb_in1.reserve_back` → `receiver_sem.set(INVALID)` (L127) → **signal-back** `sender_sem.up(sender_x,sender_y,1)` (L130) → **receiver-wait** `wait(VALID)` (L133) → `push_back`.
- in3/bias block (FUSE_BIAS, L138-155): identical second receiver block — `set(INVALID)` (L145) → `up` (L148) → `wait(VALID)` (L151) → `cb_in3.push_back`.
- L157+: **WRITER portion** — out stores via `noc.async_write` + `async_write_barrier` + cb_out pops. OUT OF FAMILY (plain unicast drain).

## HOLEs
- None in the receiver block. Writer half is out-of-family (do not wrap). Same "reader-receiver + writer in one file" co-residence cost as the sender sibling.
