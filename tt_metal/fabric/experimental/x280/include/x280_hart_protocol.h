// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Per-hart resident-idle lifecycle protocol shared by host code, firmware,
 * and preprocessed assembly.
 */
#ifndef X280_HART_PROTOCOL_H
#define X280_HART_PROTOCOL_H

#include "x280_address_map.h"

/* Protocol placement and record geometry. */
#define X280_HART_PROTOCOL_VERSION X280_ABI_U32(0x00010000)
#define X280_HART_COUNT X280_ABI_U32(4)
#define X280_HART_RECORD_SHIFT X280_ABI_U32(6)
#define X280_HART_RECORD_STRIDE X280_ABI_U64(0x40)
#define X280_HART_PROTOCOL_BASE_OFFSET X280_ABI_U64(0x1800)
#define X280_HART_PROTOCOL_LIM_OFFSET X280_ABI_U64(0x00131800)
#define X280_HART_PROTOCOL_BASE (X280_LIM_BASE + X280_HART_PROTOCOL_LIM_OFFSET)
#define X280_HART_PROTOCOL_SIZE X280_ABI_U64(0x300)
#define X280_HART_PROTOCOL_END_OFFSET (X280_HART_PROTOCOL_BASE_OFFSET + X280_HART_PROTOCOL_SIZE)
/* One past the protocol block; the host runtime reserves LIM below it. */
#define X280_HART_PROTOCOL_END (X280_HART_PROTOCOL_BASE + X280_HART_PROTOCOL_SIZE)

/* Record groups within x280_hart_protocol_t. */
#define X280_HART_CONTROL_BASE_OFFSET X280_ABI_U64(0x000)
#define X280_HART_STATE_BASE_OFFSET X280_ABI_U64(0x100)
#define X280_HART_TRAP_BASE_OFFSET X280_ABI_U64(0x200)

/* x280_hart_control_t fields. */
#define X280_HART_CONTROL_RUN_VALUE_OFFSET X280_ABI_U64(0x00)
#define X280_HART_CONTROL_ENTRY_OFFSET X280_ABI_U64(0x08)
#define X280_HART_CONTROL_COMMAND_OFFSET X280_ABI_U64(0x10)
#define X280_HART_CONTROL_ARGS_OFFSET X280_ABI_U64(0x18)
#define X280_HART_CONTROL_ARGS_BYTES_OFFSET X280_ABI_U64(0x20)

/* x280_hart_state_t fields. */
#define X280_HART_STATE_PROTOCOL_VERSION_OFFSET X280_ABI_U64(0x00)
#define X280_HART_STATE_TERMINAL_STATUS_OFFSET X280_ABI_U64(0x04)
#define X280_HART_STATE_RUN_VALUE_OFFSET X280_ABI_U64(0x08)
#define X280_HART_STATE_PHASE_OFFSET X280_ABI_U64(0x10)
#define X280_HART_STATE_RESULT_OFFSET X280_ABI_U64(0x18)
#define X280_HART_STATE_HEARTBEAT_OFFSET X280_ABI_U64(0x20)
#define X280_HART_STATE_STAGE_OFFSET X280_ABI_U64(0x28)

/* x280_hart_trap_t fields. */
#define X280_HART_TRAP_RUN_VALUE_OFFSET X280_ABI_U64(0x00)
#define X280_HART_TRAP_MCAUSE_OFFSET X280_ABI_U64(0x08)
#define X280_HART_TRAP_MEPC_OFFSET X280_ABI_U64(0x10)
#define X280_HART_TRAP_MTVAL_OFFSET X280_ABI_U64(0x18)
#define X280_HART_TRAP_PROBE_STATE_OFFSET X280_ABI_U64(0x20)
#define X280_HART_TRAP_VALID_OFFSET X280_ABI_U64(0x28)

#define X280_HART_CONTROL_ADDR(hart) \
    (X280_HART_PROTOCOL_BASE + X280_HART_CONTROL_BASE_OFFSET + (hart) * X280_HART_RECORD_STRIDE)
#define X280_HART_STATE_ADDR(hart) \
    (X280_HART_PROTOCOL_BASE + X280_HART_STATE_BASE_OFFSET + (hart) * X280_HART_RECORD_STRIDE)
#define X280_HART_TRAP_ADDR(hart) \
    (X280_HART_PROTOCOL_BASE + X280_HART_TRAP_BASE_OFFSET + (hart) * X280_HART_RECORD_STRIDE)

/* Host-to-hart commands. */
#define X280_HART_COMMAND_NONE X280_ABI_U64(0x0000000000000000)
#define X280_HART_COMMAND_JUMP X280_ABI_U64(0x000000004A554D50)
#define X280_HART_COMMAND_STOP X280_ABI_U64(0x0000000053544F50)
#define X280_HART_COMMAND_RETURN_ACK X280_ABI_U64(0x0000000041434B21)

/* Hart-to-host phases and terminal status. */
#define X280_HART_PHASE_IDLE X280_ABI_U64(0x000000001D1E0001)
#define X280_HART_PHASE_RUNNING_ACTIVE_FW X280_ABI_U64(0x000000007E570001)
#define X280_HART_PHASE_RETURNED_TO_IDLE X280_ABI_U64(0x000000001D1E0002)
#define X280_HART_STATUS_SUCCESS X280_ABI_U32(0x10000003)

/* How far the lifecycle harness (src/x280_active_start.c) has carried this
 * hart. A free-running postcode, not a commit word: no host predicate reads
 * it. Enqueue's record clear leaves it at 0 until the harness runs, so 0
 * means the hart never entered. */
#define X280_HART_STAGE_ENTERED X280_ABI_U32(1)
#define X280_HART_STAGE_MAIN X280_ABI_U32(2)
#define X280_HART_STAGE_POSTED X280_ABI_U32(3)
#define X280_HART_STAGE_STOPPING X280_ABI_U32(4)
#define X280_HART_STAGE_RETURNING X280_ABI_U32(5)

#define X280_HART_IDLE_HEARTBEAT_BASE X280_ABI_U64(0x1D1E100000000000)
#define X280_HART_ACTIVE_HEARTBEAT_BASE X280_ABI_U64(0xAC71000000000000)
#define X280_HART_IDLE_HEARTBEAT(hart) (X280_HART_IDLE_HEARTBEAT_BASE + (hart))
#define X280_HART_ACTIVE_HEARTBEAT(run, hart) (X280_HART_ACTIVE_HEARTBEAT_BASE + (run) + (hart))

/* Machine-software-interrupt acknowledgement written by entry.S. */
#define X280_IPI_ACK_OFFSET X280_ABI_U64(0x470020)
#define X280_IPI_ACK_ADDR (X280_LOCAL_DDR_TILE_BASE + X280_IPI_ACK_OFFSET)
#define X280_IPI_ACK_RECORD_SHIFT X280_ABI_U32(3)
#define X280_IPI_ACK_FLAG X280_ABI_U64(0xACEC000000000001)

#ifndef __ASSEMBLER__

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define X280_WIRE_ASSERT static_assert
#define X280_WIRE_ALIGNOF alignof
#else
#define X280_WIRE_ASSERT _Static_assert
#define X280_WIRE_ALIGNOF _Alignof
#endif

/* args addresses a caller-owned block in the tile's available LIM, or is zero
 * when the image takes no arguments. args_bytes is the validated extent. The
 * runtime validates placement, not freshness or contents, so each image must
 * validate its own layout marker. */
typedef struct __attribute__((aligned(X280_HART_RECORD_STRIDE))) {
    uint64_t run_value;
    uint64_t entry;
    uint64_t command;
    uint64_t args;
    uint64_t args_bytes;
    uint8_t reserved[24];
} x280_hart_control_t;

typedef struct __attribute__((aligned(X280_HART_RECORD_STRIDE))) {
    uint32_t protocol_version;
    uint32_t terminal_status;
    uint64_t run_value;
    uint64_t phase;
    uint64_t result;
    uint64_t heartbeat;
    uint32_t stage;
    uint32_t reserved_stage;
    uint8_t reserved[16];
} x280_hart_state_t;

typedef struct __attribute__((aligned(X280_HART_RECORD_STRIDE))) {
    uint64_t run_value;
    uint64_t mcause;
    uint64_t mepc;
    uint64_t mtval;
    uint64_t probe_state;
    uint64_t trap_valid;
    uint8_t reserved[16];
} x280_hart_trap_t;

/* Arrays of record types preserve each assembly-visible group base. */
typedef struct __attribute__((aligned(X280_HART_RECORD_STRIDE))) {
    x280_hart_control_t control[X280_HART_COUNT];
    x280_hart_state_t state[X280_HART_COUNT];
    x280_hart_trap_t trap[X280_HART_COUNT];
} x280_hart_protocol_t;

X280_WIRE_ASSERT(sizeof(x280_hart_control_t) == X280_HART_RECORD_STRIDE, "hart control record stride");
X280_WIRE_ASSERT((1ULL << X280_HART_RECORD_SHIFT) == X280_HART_RECORD_STRIDE, "hart record shift");
X280_WIRE_ASSERT(X280_WIRE_ALIGNOF(x280_hart_control_t) == X280_HART_RECORD_STRIDE, "hart control record alignment");
X280_WIRE_ASSERT(
    offsetof(x280_hart_control_t, run_value) == X280_HART_CONTROL_RUN_VALUE_OFFSET, "hart control run-value offset");
X280_WIRE_ASSERT(offsetof(x280_hart_control_t, entry) == X280_HART_CONTROL_ENTRY_OFFSET, "hart control entry offset");
X280_WIRE_ASSERT(
    offsetof(x280_hart_control_t, command) == X280_HART_CONTROL_COMMAND_OFFSET, "hart control command offset");
X280_WIRE_ASSERT(offsetof(x280_hart_control_t, args) == X280_HART_CONTROL_ARGS_OFFSET, "hart control args offset");
X280_WIRE_ASSERT(
    offsetof(x280_hart_control_t, args_bytes) == X280_HART_CONTROL_ARGS_BYTES_OFFSET, "hart control args-bytes offset");

X280_WIRE_ASSERT(sizeof(x280_hart_state_t) == X280_HART_RECORD_STRIDE, "hart state record stride");
X280_WIRE_ASSERT(X280_WIRE_ALIGNOF(x280_hart_state_t) == X280_HART_RECORD_STRIDE, "hart state record alignment");
X280_WIRE_ASSERT(
    offsetof(x280_hart_state_t, protocol_version) == X280_HART_STATE_PROTOCOL_VERSION_OFFSET,
    "hart state version offset");
X280_WIRE_ASSERT(
    offsetof(x280_hart_state_t, terminal_status) == X280_HART_STATE_TERMINAL_STATUS_OFFSET, "hart state status offset");
X280_WIRE_ASSERT(
    offsetof(x280_hart_state_t, run_value) == X280_HART_STATE_RUN_VALUE_OFFSET, "hart state run-value offset");
X280_WIRE_ASSERT(offsetof(x280_hart_state_t, phase) == X280_HART_STATE_PHASE_OFFSET, "hart state phase offset");
X280_WIRE_ASSERT(offsetof(x280_hart_state_t, result) == X280_HART_STATE_RESULT_OFFSET, "hart state result offset");
X280_WIRE_ASSERT(
    offsetof(x280_hart_state_t, heartbeat) == X280_HART_STATE_HEARTBEAT_OFFSET, "hart state heartbeat offset");
X280_WIRE_ASSERT(offsetof(x280_hart_state_t, stage) == X280_HART_STATE_STAGE_OFFSET, "hart state stage offset");

X280_WIRE_ASSERT(sizeof(x280_hart_trap_t) == X280_HART_RECORD_STRIDE, "hart trap record stride");
X280_WIRE_ASSERT(X280_WIRE_ALIGNOF(x280_hart_trap_t) == X280_HART_RECORD_STRIDE, "hart trap record alignment");
X280_WIRE_ASSERT(
    offsetof(x280_hart_trap_t, run_value) == X280_HART_TRAP_RUN_VALUE_OFFSET, "hart trap run-value offset");
X280_WIRE_ASSERT(offsetof(x280_hart_trap_t, mcause) == X280_HART_TRAP_MCAUSE_OFFSET, "hart trap mcause offset");
X280_WIRE_ASSERT(offsetof(x280_hart_trap_t, mepc) == X280_HART_TRAP_MEPC_OFFSET, "hart trap mepc offset");
X280_WIRE_ASSERT(offsetof(x280_hart_trap_t, mtval) == X280_HART_TRAP_MTVAL_OFFSET, "hart trap mtval offset");
X280_WIRE_ASSERT(
    offsetof(x280_hart_trap_t, probe_state) == X280_HART_TRAP_PROBE_STATE_OFFSET, "hart trap probe-state offset");
X280_WIRE_ASSERT(offsetof(x280_hart_trap_t, trap_valid) == X280_HART_TRAP_VALID_OFFSET, "hart trap valid offset");

X280_WIRE_ASSERT(sizeof(x280_hart_protocol_t) == X280_HART_PROTOCOL_SIZE, "hart protocol region size");
X280_WIRE_ASSERT(X280_WIRE_ALIGNOF(x280_hart_protocol_t) == X280_HART_RECORD_STRIDE, "hart protocol region alignment");
X280_WIRE_ASSERT(offsetof(x280_hart_protocol_t, control) == X280_HART_CONTROL_BASE_OFFSET, "hart control group offset");
X280_WIRE_ASSERT(offsetof(x280_hart_protocol_t, state) == X280_HART_STATE_BASE_OFFSET, "hart state group offset");
X280_WIRE_ASSERT(offsetof(x280_hart_protocol_t, trap) == X280_HART_TRAP_BASE_OFFSET, "hart trap group offset");

X280_WIRE_ASSERT(X280_HART_COMMAND_NONE != X280_HART_COMMAND_JUMP, "hart commands must be distinct");
X280_WIRE_ASSERT(X280_HART_COMMAND_JUMP != X280_HART_COMMAND_STOP, "hart commands must be distinct");
X280_WIRE_ASSERT(X280_HART_COMMAND_STOP != X280_HART_COMMAND_RETURN_ACK, "hart commands must be distinct");
X280_WIRE_ASSERT(
    X280_HART_IDLE_HEARTBEAT_BASE != X280_HART_ACTIVE_HEARTBEAT_BASE, "hart heartbeat domains must be distinct");

X280_WIRE_ASSERT(X280_HART_CONTROL_ADDR(0) == 0x08131800, "hart control assembly address");
X280_WIRE_ASSERT(X280_HART_TRAP_ADDR(0) == 0x08131A00, "hart trap assembly address");
X280_WIRE_ASSERT(
    X280_HART_TRAP_ADDR(0) + X280_HART_TRAP_PROBE_STATE_OFFSET == 0x08131A20, "hart probe-state assembly address");
X280_WIRE_ASSERT(X280_IPI_ACK_ADDR == 0x400030470020, "IPI acknowledgement assembly address");
X280_WIRE_ASSERT((1ULL << X280_IPI_ACK_RECORD_SHIFT) == sizeof(uint64_t), "IPI acknowledgement record shift");
X280_WIRE_ASSERT(
    X280_IDLE_TRAP_DIAG_ADDR + X280_HART_COUNT * X280_TRAP_DIAG_STRIDE ==
        X280_IDLE_FW_LOAD_ADDR + X280_IDLE_FW_REGION_SIZE,
    "idle trap blocks must end at the idle region end");
X280_WIRE_ASSERT(
    X280_ACTIVE_TRAP_DIAG_ADDR + X280_HART_COUNT * X280_TRAP_DIAG_STRIDE == X280_ACTIVE_FW_REGION_END,
    "active trap blocks must end at the active region end");

#undef X280_WIRE_ASSERT
#undef X280_WIRE_ALIGNOF

#endif /* !__ASSEMBLER__ */

#endif /* X280_HART_PROTOCOL_H */
