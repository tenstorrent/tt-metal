# Kernel Execution Trace: add_2_integers_in_riscv

This document traces the execution path from `kernel_main()` down to hardware register writes, with exact file paths and line numbers.

## Entry Point: Kernel Invocation

**Kernel Function:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:5`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L5)
  - `void kernel_main()` - Entry point for the kernel

**Called From:**
- [`tt_metal/hw/firmware/src/tt-1xx/brisck.cc:74`](tt_metal/hw/firmware/src/tt-1xx/brisck.cc#L74)
  - `kernel_main();` - Called from BRISC (RISCV_0) entry point
- [`tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc:54`](tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc#L54)
  - `do_crt1()` - Runtime initialization that eventually calls `kernel_main()`

---

## 1. Reading Runtime Arguments

### 1.1 get_arg_val() - Read Runtime Argument

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:6-11`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L6-L11)
  - `uint32_t src0_dram = get_arg_val<uint32_t>(0);`
  - `uint32_t src1_dram = get_arg_val<uint32_t>(1);`
  - `uint32_t dst_dram = get_arg_val<uint32_t>(2);`
  - `uint32_t src0_l1 = get_arg_val<uint32_t>(3);`
  - `uint32_t src1_l1 = get_arg_val<uint32_t>(4);`
  - `uint32_t dst_l1 = get_arg_val<uint32_t>(5);`

**Implementation:**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:135-139`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L135-L139)
  - `template <typename T> FORCE_INLINE T get_arg_val(int arg_idx)`
  - Calls `get_arg_addr(arg_idx)` and dereferences the pointer

**Helper Function:**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:105`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L105)
  - `static FORCE_INLINE uintptr_t get_arg_addr(int arg_idx)`
  - Returns `&rta_l1_base[arg_idx]` - address of runtime argument in L1 memory

**ABI Level:**
- `rta_l1_base` is a global pointer to L1 memory where runtime arguments are stored
- The host writes these values to L1 before kernel execution
- Reading from L1 memory is a simple memory load instruction (RISC-V `lw`)

---

## 2. Creating Address Generators

### 2.1 InterleavedAddrGen Structure

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:16-18`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L16-L18)
  - `InterleavedAddrGen<true> src0 = {.bank_base_address = src0_dram, .page_size = sizeof(uint32_t)};`
  - `InterleavedAddrGen<true> src1 = {.bank_base_address = src1_dram, .page_size = sizeof(uint32_t)};`
  - `InterleavedAddrGen<true> dst = {.bank_base_address = dst_dram, .page_size = sizeof(uint32_t)};`

**Structure Definition:**
- [`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:268-295`](tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h#L268-L295)
  - `struct InterleavedAddrGen<DRAM>`
  - Fields:
    - `bank_base_address` (uint32_t) - Base address valid on all banks (lock-step allocation)
    - `page_size` (const uint32_t) - Size of each page
    - `aligned_page_size` (const uint32_t) - Page size aligned to allocator requirements

---

## 3. Calculating NOC Addresses

### 3.1 get_noc_addr() - Calculate NOC Address from Page Index

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:28-29`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L28-L29)
  - `uint64_t src0_dram_noc_addr = get_noc_addr(0, src0);`
  - `uint64_t src1_dram_noc_addr = get_noc_addr(0, src1);`

**Method Call:**
- [`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:286-294`](tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h#L286-L294)
  - `std::uint64_t InterleavedAddrGen::get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const`
  - Steps:
    1. Calculate `bank_offset_index` = `id / NUM_DRAM_BANKS`
    2. Calculate `bank_index` = `id % NUM_DRAM_BANKS` (round-robin)
    3. Calculate `addr` = base address + offset within bank
    4. Get `noc_xy` = NOC coordinates for the DRAM bank
    5. Encode into 64-bit NOC address

**Helper Functions:**

**Bank Offset Calculation:**
- [`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:18-33`](tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h#L18-L33)
  - `get_bank_offset_index<DRAM>(uint32_t id)` - Calculates which "round" of pages (id / NUM_BANKS)

**Bank Index Calculation:**
- [`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:35-42`](tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h#L35-L42)
  - `get_bank_index<DRAM>(uint32_t id, uint32_t bank_offset_index)` - Calculates which bank (id % NUM_BANKS)

**NOC Coordinates Lookup:**
- [`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:44-51`](tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h#L44-L51)
  - `get_noc_xy<DRAM>(uint32_t bank_index, uint8_t noc)` - Looks up NOC (x,y) coordinates from `dram_bank_to_noc_xy[noc][bank_index]`
  - This table is initialized at device startup and stored in L1 memory

**Address Encoding:**
- [`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:205-211`](tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h#L205-L211)
  - `get_noc_addr_helper(uint32_t noc_xy, uint32_t addr)` - Encodes NOC address
  - Returns: `((uint64_t)(noc_xy) << NOC_ADDR_COORD_SHIFT) | addr`
  - Encodes destination core coordinates (upper bits) + local address (lower 32 bits)

---

## 4. Reading from DRAM

### 4.1 noc_async_read() - Initiate Asynchronous Read

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:30-31`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L30-L31)
  - `noc_async_read(src0_dram_noc_addr, src0_l1, sizeof(uint32_t));`
  - `noc_async_read(src1_dram_noc_addr, src1_l1, sizeof(uint32_t));`

**API Implementation:**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:532-554`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L532-L554)
  - `inline void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc = noc_index, uint32_t read_req_vc = NOC_UNICAST_WRITE_VC)`
  - For small transfers (≤ NOC_MAX_BURST_SIZE): calls `noc_async_read_one_packet()`
  - For large transfers: calls `ncrisc_noc_fast_read_any_len()`

**Lower-Level Implementation (Small Transfer):**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L547)
  - `noc_async_read_one_packet<false>(src_noc_addr, dst_local_l1_addr, size, noc, read_req_vc);`

**Lower-Level Implementation (Large Transfer):**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:551`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L551)
  - `ncrisc_noc_fast_read_any_len<noc_mode>(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size, read_req_vc);`

**NOC Fast Read (Any Length):**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:515-531`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L515-L531)
  - `ncrisc_noc_fast_read_any_len()` - Splits large transfers into multiple packets
  - Loops calling `ncrisc_noc_fast_read()` for each packet

**NOC Fast Read (Single Packet):**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:168-191`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L168-L191)
  - `ncrisc_noc_fast_read()` - Writes to NOC command buffer registers:
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr)` - Return address (L1)
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr)` - Target address low
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT))` - Target coordinates
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes)` - Transfer length
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ)` - **Trigger the transfer**

**Hardware Register Write Macro:**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:141-159`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L141-L159)
  - `NOC_CMD_BUF_WRITE_REG(noc, buf, addr, val)` - Writes to memory-mapped NOC registers
  - Calculates offset: `(buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr`
  - Writes: `*((volatile uint32_t*)offset) = val;`
  - **ABI Level:** This is a memory-mapped I/O write - the hardware NOC controller responds to these register writes

### 4.2 noc_async_read_barrier() - Wait for Read Completion

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:32`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L32)
  - `noc_async_read_barrier();`

**Implementation:**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:1575-1590`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L1575-L1590)
  - `void noc_async_read_barrier(uint8_t noc = noc_index)`
  - Polls `ncrisc_noc_reads_flushed(noc)` until all reads complete
  - Calls `invalidate_l1_cache()` to ensure data is visible

**Read Flush Check:**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:200-202`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L200-L202)
  - `ncrisc_noc_reads_flushed(uint32_t noc)` - Checks if all reads are complete
  - Compares: `NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]`

**Status Register Read:**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:160-163`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L160-L163)
  - `NOC_STATUS_READ_REG(noc, reg_id)` - Reads NOC status register
  - Calculates offset: `(noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(reg_id)`
  - Returns: `*((volatile uint32_t*)offset)`
  - **ABI Level:** Memory-mapped I/O read - polls hardware status register

---

## 5. Computation (Simple Addition)

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:35-41`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L35-L41)
  - `uint32_t* dat0 = (uint32_t*)src0_l1;`
  - `uint32_t* dat1 = (uint32_t*)src1_l1;`
  - `uint32_t* out0 = (uint32_t*)dst_l1;`
  - `(*out0) = (*dat0) + (*dat1);`

**ABI Level:**
- This compiles to standard RISC-V instructions:
  - `lw` (load word) - Load values from L1 memory
  - `add` - Add two 32-bit integers
  - `sw` (store word) - Store result to L1 memory
- Executes directly on the RISC-V processor (BRISC/NCRISC)

---

## 6. Writing to DRAM

### 6.1 get_noc_addr() - Calculate Destination NOC Address

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:46`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L46)
  - `uint64_t dst_dram_noc_addr = get_noc_addr(0, dst);`

**Same as Section 3.1** - Uses the same `InterleavedAddrGen::get_noc_addr()` method

### 6.2 noc_async_write() - Initiate Asynchronous Write

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:47`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L47)
  - `noc_async_write(dst_l1, dst_dram_noc_addr, sizeof(uint32_t));`

**API Implementation:**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:798-817`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L798-L817)
  - `inline void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size, uint8_t noc = noc_index, uint32_t vc = NOC_UNICAST_WRITE_VC)`
  - For small transfers: calls `noc_async_write_one_packet()`
  - For large transfers: calls `ncrisc_noc_fast_write_any_len()`

**NOC Fast Write (Any Length):**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:534-546`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L534-L546)
  - `ncrisc_noc_fast_write_any_len()` - Splits large transfers into multiple packets
  - Loops calling `ncrisc_noc_fast_write()` for each packet

**NOC Fast Write (Single Packet):**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:214-244`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L214-L244)
  - `ncrisc_noc_fast_write()` - Writes to NOC command buffer registers:
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, src_addr)` - Source address (L1)
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)dest_addr)` - Target address low
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT))` - Target coordinates
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes)` - Transfer length
    - `NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ)` - **Trigger the transfer**

**Hardware Register Write:**
- Same as Section 4.1 - Uses `NOC_CMD_BUF_WRITE_REG()` macro

### 6.3 noc_async_write_barrier() - Wait for Write Completion

**Kernel Code:**
- [`tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:48`](tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp#L48)
  - `noc_async_write_barrier();`

**Implementation:**
- [`tt_metal/hw/inc/api/dataflow/dataflow_api.h:1605-1620`](tt_metal/hw/inc/api/dataflow/dataflow_api.h#L1605-L1620)
  - `void noc_async_write_barrier(uint8_t noc = noc_index)`
  - Polls `ncrisc_noc_nonposted_writes_flushed(noc)` until all writes complete
  - Calls `invalidate_l1_cache()` to ensure coherency

**Write Flush Check:**
- [`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:261-263`](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h#L261-L263)
  - `ncrisc_noc_nonposted_writes_flushed(uint32_t noc)` - Checks if all writes are complete
  - Compares: `NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_num_issued[noc]`

**Status Register Read:**
- Same as Section 4.2 - Uses `NOC_STATUS_READ_REG()` macro

---

## Summary: Execution Flow

```
kernel_main() [Kernel Entry]
  ↓
get_arg_val() [Read runtime args from L1]
  ↓
InterleavedAddrGen{} [Create address generator struct]
  ↓
get_noc_addr() [Calculate NOC address from page index]
  ├─→ get_bank_offset_index() [Calculate which round of pages]
  ├─→ get_bank_index() [Calculate which bank (round-robin)]
  ├─→ get_noc_xy() [Lookup NOC coordinates from table]
  └─→ get_noc_addr_helper() [Encode NOC address]
  ↓
noc_async_read() [Initiate DRAM read]
  ├─→ ncrisc_noc_fast_read_any_len() [Split into packets]
  │   └─→ ncrisc_noc_fast_read() [Write NOC registers]
  │       └─→ NOC_CMD_BUF_WRITE_REG() [Memory-mapped I/O]
  └─→ [Hardware NOC executes transfer]
  ↓
noc_async_read_barrier() [Wait for completion]
  ├─→ ncrisc_noc_reads_flushed() [Check status]
  │   └─→ NOC_STATUS_READ_REG() [Poll hardware register]
  └─→ invalidate_l1_cache() [Ensure coherency]
  ↓
[Simple Addition] [RISC-V add instruction]
  ↓
noc_async_write() [Initiate DRAM write]
  ├─→ ncrisc_noc_fast_write_any_len() [Split into packets]
  │   └─→ ncrisc_noc_fast_write() [Write NOC registers]
  │       └─→ NOC_CMD_BUF_WRITE_REG() [Memory-mapped I/O]
  └─→ [Hardware NOC executes transfer]
  ↓
noc_async_write_barrier() [Wait for completion]
  ├─→ ncrisc_noc_nonposted_writes_flushed() [Check status]
  │   └─→ NOC_STATUS_READ_REG() [Poll hardware register]
  └─→ invalidate_l1_cache() [Ensure coherency]
```

---

## Key Concepts

1. **Lock-Step Allocation**: All DRAM banks reserve the same address space, so `bank_base_address` is valid on all banks
2. **Round-Robin Interleaving**: Pages are distributed across banks using `page_index % NUM_BANKS`
3. **NOC Addressing**: 64-bit address encodes destination core (x,y) in upper bits + local address in lower 32 bits
4. **Memory-Mapped I/O**: NOC registers are accessed via memory-mapped addresses - writes trigger hardware actions
5. **Asynchronous Operations**: Reads/writes return immediately; barriers wait for completion by polling status registers
