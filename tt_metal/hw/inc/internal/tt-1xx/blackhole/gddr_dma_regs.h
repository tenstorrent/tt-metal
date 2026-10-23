// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hardware register map for the DRISC GDDR DMA engine — do not include directly.

#ifdef COMPILE_FOR_DRISC

#include <stdint.h>

// TX Stream base registers
#define TX_REG_STREAM_REG_MAP_BASE_ADDR (0xFC000000)
#define TX_REG_SIZE (0x100)

// TX CTRL base register
#define TX_CTRL_REG_MAP_BASE_ADDR (0xFC001000)

#define TX_CTRL_TX_CLK_CTRL_REG_OFFSET (0x00000000)
#define TX_CTRL_TX_TRANSFER_ATTRIBUTES_REG_OFFSET (0x00000004)
#define TX_CTRL_TX_FLOW_CONTROL_REG_OFFSET (0x00000008)
#define TX_CTRL_TX_REMOTE_UPDATE_STREAM_BASE_REG_OFFSET (0x0000000C)
#define TX_CTRL_TX_READ_STATUS_REG_OFFSET (0x00000010)
#define TX_CTRL_TX_WRITE_STATUS_REG_OFFSET (0x00000014)
#define TX_CTRL_TX_INTERRUPT_CTRL_REG_OFFSET (0x00000150)

// TX Stream registers
#define TX_REG_STREAM_STATUS_REG_OFFSET (0x00000000)
#define TX_REG_STREAM_TXQ_SETTINGS_REG_OFFSET (0x00000004)
#define TX_REG_STREAM_PAUSE_STATUS_REG_OFFSET (0x00000008)
#define TX_REG_STREAM_WRITE_TRANSFER_START_ADDR_REG_OFFSET (0x00000010)
#define TX_REG_STREAM_WRITE_DEST_ADDR_LOW_REG_OFFSET (0x00000014)
#define TX_REG_STREAM_WRITE_DEST_ADDR_HIGH_REG_OFFSET (0x00000018)
#define TX_REG_STREAM_WRITE_TILE_HEADER_DEST_ADDR_REG_OFFSET (0x0000001C)
#define TX_REG_STREAM_WRITE_STREAM_ID_REG_OFFSET (0x00000020)
#define TX_REG_STREAM_REMOTE_REG_DATA_REG_OFFSET (0x00000024)
#define TX_REG_STREAM_READ_TRANSFER_SOURCE_LOW_REG_OFFSET (0x00000030)
#define TX_REG_STREAM_READ_TRANSFER_SOURCE_HIGH_REG_OFFSET (0x00000034)
#define TX_REG_STREAM_READ_TRANSFER_DEST_REG_OFFSET (0x00000038)
#define TX_REG_STREAM_WRITE_AUTO_INCR_SRC_OFFSET_REG_OFFSET (0x00000040)
#define TX_REG_STREAM_WRITE_AUTO_INCR_DEST_OFFSET_REG_OFFSET (0x00000044)
#define TX_REG_STREAM_READ_AUTO_INCR_SRC_OFFSET_REG_OFFSET (0x00000048)
#define TX_REG_STREAM_READ_AUTO_INCR_DEST_OFFSET_REG_OFFSET (0x0000004C)
#define TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET (0x00000050)
#define TX_REG_STREAM_SCRATCH_0_REG_OFFSET (0x00000058)
#define TX_REG_STREAM_SCRATCH_1_REG_OFFSET (0x0000005C)
#define TX_REG_STREAM_READ_AUTO_INC_CUR_SRC_LOW_REG_OFFSET (0x00000060)
#define TX_REG_STREAM_READ_AUTO_INC_CUR_SRC_HIGH_REG_OFFSET (0x00000064)
#define TX_REG_STREAM_READ_AUTO_INC_DEST_REG_OFFSET (0x00000068)
#define TX_REG_STREAM_WRITE_AUTO_INC_CUR_SRC_REG_OFFSET (0x0000006C)
#define TX_REG_STREAM_WRITE_AUTO_INC_CUR_DEST_LOW_REG_OFFSET (0x00000070)
#define TX_REG_STREAM_WRITE_AUTO_INC_CUR_DEST_HIGH_REG_OFFSET (0x00000074)
#define TX_REG_STREAM_TOTAL_WRITES_REG_OFFSET (0x00000078)
#define TX_REG_STREAM_TOTAL_READS_REG_OFFSET (0x0000007C)

#define READ_TX_CTRL_REG(offset) (*(volatile uint32_t*)(TX_CTRL_REG_MAP_BASE_ADDR + (offset)))

#define WRITE_TX_CTRL_REG(offset, val) (*(volatile uint32_t*)(TX_CTRL_REG_MAP_BASE_ADDR + (offset)) = (val))

#define READ_TX_STREAM_REG(stream, offset) \
    (*(volatile uint32_t*)(TX_REG_STREAM_REG_MAP_BASE_ADDR + TX_REG_SIZE * (stream) + (offset)))

#define WRITE_TX_STREAM_REG(stream, offset, val) \
    (*(volatile uint32_t*)(TX_REG_STREAM_REG_MAP_BASE_ADDR + TX_REG_SIZE * (stream) + (offset)) = (val))

// TX Queue Status Register
typedef struct {
    uint32_t has_outstanding_write : 1;
    uint32_t has_outstanding_remote_update : 1;
    uint32_t has_outstanding_data_forward : 1;
    uint32_t has_outstanding_raw_write : 1;
    uint32_t has_outstanding_read : 1;
    uint32_t rsvd_0 : 3;
    uint32_t num_reads_outstanding : 8;
    uint32_t num_writes_outstanding : 4;
    uint32_t num_remote_updates_outstanding : 4;
    uint32_t num_data_forwards_outstanding : 4;
    uint32_t num_raw_writes_outstanding : 4;
} DmaTxqStatus_t;

typedef union {
    uint32_t val;
    DmaTxqStatus_t f;
} DmaTxqStatus_u;

#define DmaTxqStatus_DEFAULT (0x00000000u)

// Masks for per-direction barrier polling on DmaTxqStatus_u.val
#define DmaTxqStatus_READ_MASK (0x0000FF10u)  // has_outstanding_read | num_reads_outstanding
#define DmaTxqStatus_WRITE_MASK \
    (0xF00F0009u)  // has_outstanding_write | has_outstanding_raw_write | num_writes_outstanding |
                   // num_raw_writes_outstanding

// TX Queue Settings Register
typedef struct {
    uint32_t stream_driven_mode : 1;
    uint32_t rsvd_0 : 3;
    uint32_t coarse_grain_auto_incr_en : 1;
} DmaTxqSettings_t;

typedef union {
    uint32_t val;
    DmaTxqSettings_t f;
} DmaTxqSettings_u;

#define DmaTxqSettings_DEFAULT (0x00000001u)

// Transfer Attributes Register
typedef struct {
    uint32_t transfer_size_words : 14;
    uint32_t rsvd_0 : 6;
    uint32_t write_auto_increment_src_addr : 1;
    uint32_t write_auto_increment_dst_addr : 1;
    uint32_t read_auto_increment_src_addr : 1;
    uint32_t read_auto_increment_dst_addr : 1;
    uint32_t start_of_packet : 1;
    uint32_t end_of_packet : 1;
    uint32_t trigger_interrupt : 1;
    uint32_t rsvd_1 : 1;
    uint32_t transfer_start_raw : 1;
    uint32_t transfer_start_data : 1;
    uint32_t transfer_start_reg : 1;
    uint32_t transfer_start_read : 1;
} DmaTxqTransferAttrs_t;

typedef union {
    uint32_t val;
    DmaTxqTransferAttrs_t f;
} DmaTxqTransferAttrs_u;

#define DmaTxqTransferAttrs_DEFAULT (0x03000000u)

// TX Read Status Register
typedef struct {
    uint32_t read_ready : 1;
    uint32_t read_busy : 1;
    uint32_t rsvd_0 : 6;
    uint32_t total_read_outstanding : 8;
} DmaCtrlReadStatus_t;

typedef union {
    uint32_t val;
    DmaCtrlReadStatus_t f;
} DmaCtrlReadStatus_u;

#define DmaCtrlReadStatus_DEFAULT (0x00000001u)

// TX Write Status Register
typedef struct {
    uint32_t write_ready : 1;
    uint32_t write_busy : 1;
    uint32_t rsvd_0 : 6;
    uint32_t remote_update_outstanding : 1;
    uint32_t data_forward_outstanding : 1;
    uint32_t raw_write_outstanding : 1;
    uint32_t rsvd_1 : 1;
    uint32_t total_write_outstanding : 4;
    uint32_t total_num_remote_update_outstanding : 4;
    uint32_t total_num_data_forward_outstanding : 4;
    uint32_t total_num_raw_write_outstanding : 4;
} DmaCtrlWriteStatus_t;

typedef union {
    uint32_t val;
    DmaCtrlWriteStatus_t f;
} DmaCtrlWriteStatus_u;

#define DmaCtrlWriteStatus_DEFAULT (0x00000001u)

// DMA Global Transfer Attributes Register
typedef struct {
    uint32_t dma_enable : 1;
    uint32_t rsvd_0 : 3;
    uint32_t posted_write_mode_enable : 1;
    uint32_t rsvd_1 : 3;
    uint32_t max_burst_size : 8;
    uint32_t dma_axi_cache : 4;
    uint32_t dma_axi_lock : 1;
} DmaCtrlTransferAttrs_t;

typedef union {
    uint32_t val;
    DmaCtrlTransferAttrs_t f;
} DmaCtrlTransferAttrs_u;

#define DmaCtrlTransferAttrs_DEFAULT (0x00031001u)

#endif  // COMPILE_FOR_DRISC
