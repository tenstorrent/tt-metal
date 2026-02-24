after asm # done sending header -> j to .L46 which corresponds to the header send

noc_async_write_one_packet -> LBE1909



flow & ILP considerations:

### Wait for empty write slot
- waits till edm has space for packet, busy-wait on comparator of enough slots

### Send payload (non-blocking)
- compute dest buffer slot noc addr: uses bitwise manip to return noc addr
- `send_chunk_from_address`
    - `ncrisc_noc_fast_write`: `noc_nonposted_writes_num_issued` & `noc_nonposted_writes_acked[noc]` updated here
    - send write noc while bytes avail. Mandatory bookeeping for src/dst/len bytes

### Send header (non-blocking)
- compute dest buffer slot noc addr -> same busy-wait on comparator
- `send_chunk_from_address`
    - `noc_async_write_one_packet`
        - `ncrisc_noc_fast_write`: same
- `post_send_payload_increment_pointers`
    - `advance_buffer_slot_write_index`:
        - update buffer slot write cnter
        - update `buffer_slot_index`
        - update `buffer_addr`
- `update_edm_buffer_free_slots`
    - `noc_inline_dw_write_with_state`: bunch of `sw`'s
         - `noc_fast_default_write_dw_inline`: updates edm and sends req? -> necessary for EDM kernel
         - Contains updates to issued/acked as well
