// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AUTO_GENERATED! DO NOT MODIFY!                                                                                              //
//                                                                                                                             //
// Please run                                                                                                                  //
//                                                                                                                             //
// (echo '<% type=:svh_header %>' && cat noc_overlay_parameters.erb) | erb -T - > ../rtl/overlay/tt_noc_overlay_params.svh   //
// (echo '<% type=:c_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.h                     //
// (echo '<% type=:cpp_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.hpp                 //
// (echo '<% type=:rb_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.rb                   //
// Open noc_overlay_parameters.hpp and move static class varaible definitions to noc_overlay_parameters.cpp                    //
// overriding existing ones.                                                                                                   //
//                                                                                                                             //
// to regenerate                                                                                                               //                                                                                                    //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef NOC_OVERLAY_PARAMETERS_H
#define NOC_OVERLAY_PARAMETERS_H

#ifndef NOC_OVERLAY_PARAMETERS_BASIC_H
#define NOC_OVERLAY_PARAMETERS_BASIC_H

#define NOC_NUM_STREAMS 64
#define ETH_NOC_NUM_STREAMS 32

#define NUM_MCAST_STREAM_ID_START 0
#define NUM_MCAST_STREAM_ID_END   3
#define NUM_RECEIVER_ENDPOINT_STREAM_ID_START 4
#define NUM_RECEIVER_ENDPOINT_STREAM_ID_END   5
#define NUM_REMOTE_RECEIVER_STREAM_ID_START 0
#define NUM_REMOTE_RECEIVER_STREAM_ID_END 63
#define RECEIVER_ENDPOINT_STREAM_MSG_GROUP_SIZE 4
#define RECEIVER_ENDPOINT_STREAM_MSG_INFO_FIFO_GROUPS     4
#define NON_RECEIVER_ENDPOINT_STREAM_MSG_INFO_FIFO_GROUPS 2
#define DEST_READY_COMMON_CACHE_NUM_ENTRIES 24
#define DEST_READY_MCAST_CACHE_NUM_ENTRIES 8

#define NOC_OVERLAY_START_ADDR     0xFFB40000
#define NOC_STREAM_REG_SPACE_SIZE  0x1000

#define STREAM_REG_ADDR(stream_id, reg_id) ((NOC_OVERLAY_START_ADDR) + (((uint32_t)(stream_id))*(NOC_STREAM_REG_SPACE_SIZE)) + (((uint32_t)(reg_id)) << 2))

#define NUM_NOCS                   2
#define NOC0_REGS_START_ADDR       0xFFB20000
#define NOC1_REGS_START_ADDR       0xFFB30000

#define NCRISC_STREAM_RANGE_1_START 0
#define NCRISC_STREAM_RANGE_1_END   3
#define NCRISC_STREAM_RANGE_2_START 8
#define NCRISC_STREAM_RANGE_2_END   11
#define NCRISC_PIC_CONFIG_PHASE_DEFAULT           0

#ifdef TB_NOC

extern "C" {
#include "noc.h"
#include "noc_api_dpi.h"
}

#else

#define NOC_STREAM_WRITE_REG(stream_id, reg_id, val)  ((*((volatile uint32_t*)(STREAM_REG_ADDR(stream_id, reg_id)))) = (val))
#define NOC_STREAM_READ_REG(stream_id, reg_id)        (*((volatile uint32_t*)(STREAM_REG_ADDR(stream_id, reg_id))))

#define NOC_STREAM_WRITE_REG_FIELD(stream_id, reg_id, field, val) (NOC_STREAM_WRITE_REG(stream_id, reg_id, ((NOC_STREAM_READ_REG(stream_id, reg_id) & ~((1 << field##_WIDTH) - 1)) | ((val & ((1 << field##_WIDTH) - 1)) << field))))
#define NOC_STREAM_READ_REG_FIELD(stream_id, reg_id, field)       ((NOC_STREAM_READ_REG(stream_id, reg_id) >> field) & ((1 << field##_WIDTH) - 1))

#define NOC_WRITE_REG(addr, val) ((*((volatile uint32_t*)(addr)))) = (val)
#define NOC_READ_REG(addr)       (*((volatile uint32_t*)(addr)))

#endif


#define NOC_ID_WIDTH     6
#define STREAM_ID_WIDTH  6

#define DEST_CNT_WIDTH   6
#define NOC_NUM_WIDTH     1

#define STREAM_REG_INDEX_WIDTH 9
#define STREAM_REG_CFG_DATA_WIDTH 24

#define MEM_WORD_WIDTH 16
#define MEM_WORD_ADDR_WIDTH 17

#define MEM_WORD_BIT_OFFSET_WIDTH 7

#define MSG_INFO_BUF_SIZE_WORDS 256
#define MSG_INFO_BUF_SIZE_BITS  8
#define MSG_INFO_BUF_SIZE_POW_BITS 3
#define MSG_INFO_BUF_SIZE_WORDS_WIDTH (MSG_INFO_BUF_SIZE_BITS + 1)

#define GLOBAL_OFFSET_TABLE_SIZE 8
#define GLOBAL_OFFSET_TABLE_SIZE_WIDTH 3

#endif

// For endpoints with SOURCE_ENDPOINT == 1, this register is for firmware
// to register new message for sending.
// This updates the msg_info register structure directly, rather than writing to the message info
// buffer in memory.
// Must not be written when the message info register structure is full, or if
// there are message info entries in the memory buffer. (This would cause a race
// condition.)
#define   STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX   0
#define       SOURCE_ENDPOINT_NEW_MSG_ADDR            0
#define       SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH        MEM_WORD_ADDR_WIDTH
#define       SOURCE_ENDPOINT_NEW_MSG_SIZE            (SOURCE_ENDPOINT_NEW_MSG_ADDR+SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH)
#define       SOURCE_ENDPOINT_NEW_MSG_SIZE_WIDTH        (32-MEM_WORD_ADDR_WIDTH-1)
#define       SOURCE_ENDPOINT_NEW_MSG_LAST_TILE       (SOURCE_ENDPOINT_NEW_MSG_SIZE+SOURCE_ENDPOINT_NEW_MSG_SIZE_WIDTH)
#define       SOURCE_ENDPOINT_NEW_MSG_LAST_TILE_WIDTH   (1)

// For endpoints with SOURCE_ENDPOINT == 1, this register is for firmware
// to update the number of messages whose data & header are available in the memory buffer.
// Hardware register is incremented atomically if sending of previous messages is in progress.
#define   STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX   1
#define       SOURCE_ENDPOINT_NEW_MSGS_NUM              0
#define       SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH          12
#define       SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE       (SOURCE_ENDPOINT_NEW_MSGS_NUM+SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH)
#define       SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE_WIDTH   MEM_WORD_ADDR_WIDTH
#define       SOURCE_ENDPOINT_NEW_MSGS_LAST_TILE        (SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE+SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE_WIDTH)
#define       SOURCE_ENDPOINT_NEW_MSGS_LAST_TILE_WIDTH    1

// Registers that need to be programmed once per blob. (Can apply to multiple phases.)
//   * Phase/data forward options:
//      PHASE_AUTO_CONFIG = set to 1 for stream to fetch next phase configuration automatically.
//      PHASE_AUTO_ADVANCE = set to 1 for stream to advance to next phase automatically
//            (otherwise need to write STREAM_PHASE_ADVANCE below)
#define   STREAM_ONETIME_MISC_CFG_REG_INDEX   2
#define       PHASE_AUTO_CONFIG                       0
#define       PHASE_AUTO_CONFIG_WIDTH                   1
#define       PHASE_AUTO_ADVANCE                      (PHASE_AUTO_CONFIG+PHASE_AUTO_CONFIG_WIDTH)
#define       PHASE_AUTO_ADVANCE_WIDTH                  1
// set to one of the values (0-5) to select which VC control flow updates will be sent on
#define       REG_UPDATE_VC_REG                       (PHASE_AUTO_ADVANCE+PHASE_AUTO_ADVANCE_WIDTH)
#define       REG_UPDATE_VC_REG_WIDTH                   3
// Read index of global offset table, which will offset o_data_fwd_src_addr by entry value.
#define       GLOBAL_OFFSET_TABLE_RD_SRC_INDEX        (REG_UPDATE_VC_REG+REG_UPDATE_VC_REG_WIDTH)
#define       GLOBAL_OFFSET_TABLE_RD_SRC_INDEX_WIDTH    GLOBAL_OFFSET_TABLE_SIZE_WIDTH
// Read index of global offset table, which will offset o_data_fwd_dest_addr by entry value.
#define       GLOBAL_OFFSET_TABLE_RD_DEST_INDEX       (GLOBAL_OFFSET_TABLE_RD_SRC_INDEX+GLOBAL_OFFSET_TABLE_RD_SRC_INDEX_WIDTH)
#define       GLOBAL_OFFSET_TABLE_RD_DEST_INDEX_WIDTH   GLOBAL_OFFSET_TABLE_SIZE_WIDTH

// The ID of NOCs used for incoming and outgoing data, followed by misc. stream configuration options:
//   * Source - set exactly one of these to 1:
//        SOURCE_ENDPOINT = source is local math/packer
//        REMOTE_SOURCE = source is remote sender stream
//        LOCAL_SOURCES_CONNECTED = source is one or more local connected streams
//   * Destination - set one or zero of these to 1:
//        RECEIVER_ENDPOINT = stream is read by local unpacker/math
//        REMOTE_RECEIVER = stream forwards data to a remote destination or multicast group
//        LOCAL_RECEIVER = stream is connected to a local destination stream
//        None set = stream just stores data in a local buffer, without forwarding/clearing, and
//                   finishes the phase once all messages have been received
#define   STREAM_MISC_CFG_REG_INDEX   3
#define       INCOMING_DATA_NOC                       0
#define       INCOMING_DATA_NOC_WIDTH                   NOC_NUM_WIDTH
#define       OUTGOING_DATA_NOC                       (INCOMING_DATA_NOC+INCOMING_DATA_NOC_WIDTH)
#define       OUTGOING_DATA_NOC_WIDTH                   NOC_NUM_WIDTH
#define       REMOTE_SRC_UPDATE_NOC                   (OUTGOING_DATA_NOC+OUTGOING_DATA_NOC_WIDTH)
#define       REMOTE_SRC_UPDATE_NOC_WIDTH               NOC_NUM_WIDTH
#define       LOCAL_SOURCES_CONNECTED                 (REMOTE_SRC_UPDATE_NOC+REMOTE_SRC_UPDATE_NOC_WIDTH)
#define       LOCAL_SOURCES_CONNECTED_WIDTH             1
#define       SOURCE_ENDPOINT                         (LOCAL_SOURCES_CONNECTED+LOCAL_SOURCES_CONNECTED_WIDTH)
#define       SOURCE_ENDPOINT_WIDTH                     1
#define       REMOTE_SOURCE                           (SOURCE_ENDPOINT+SOURCE_ENDPOINT_WIDTH)
#define       REMOTE_SOURCE_WIDTH                       1
#define       RECEIVER_ENDPOINT                       (REMOTE_SOURCE+REMOTE_SOURCE_WIDTH)
#define       RECEIVER_ENDPOINT_WIDTH                   1
#define       LOCAL_RECEIVER                          (RECEIVER_ENDPOINT+RECEIVER_ENDPOINT_WIDTH)
#define       LOCAL_RECEIVER_WIDTH                      1
#define       REMOTE_RECEIVER                         (LOCAL_RECEIVER+LOCAL_RECEIVER_WIDTH)
#define       REMOTE_RECEIVER_WIDTH                     1
#define       TOKEN_MODE                              (REMOTE_RECEIVER+REMOTE_RECEIVER_WIDTH)
#define       TOKEN_MODE_WIDTH                          1
#define       COPY_MODE                               (TOKEN_MODE+TOKEN_MODE_WIDTH)
#define       COPY_MODE_WIDTH                           1
#define       NEXT_PHASE_SRC_CHANGE                   (COPY_MODE+COPY_MODE_WIDTH)
#define       NEXT_PHASE_SRC_CHANGE_WIDTH               1
#define       NEXT_PHASE_DEST_CHANGE                  (NEXT_PHASE_SRC_CHANGE+NEXT_PHASE_SRC_CHANGE_WIDTH)
#define       NEXT_PHASE_DEST_CHANGE_WIDTH              1
// set if REMOTE_SOURCE==1 and the buffer is large enough to accept full phase data without wrapping:
#define       DATA_BUF_NO_FLOW_CTRL                   (NEXT_PHASE_DEST_CHANGE+NEXT_PHASE_DEST_CHANGE_WIDTH)
#define       DATA_BUF_NO_FLOW_CTRL_WIDTH               1
// set if REMOTE_RECEIVER==1 and the destination buffer is large enough to accept full phase data without wrapping:
#define       DEST_DATA_BUF_NO_FLOW_CTRL              (DATA_BUF_NO_FLOW_CTRL+DATA_BUF_NO_FLOW_CTRL_WIDTH)
#define       DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH          1
// set if REMOTE_SOURCE==1 and you want the buffer to have wrapping:
#define       MSG_INFO_BUF_FLOW_CTRL                  (DEST_DATA_BUF_NO_FLOW_CTRL+DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH)
#define       MSG_INFO_BUF_FLOW_CTRL_WIDTH              1
// set if REMOTE_RECEIVER==1 and you want the destination buffer to have wrapping:
#define       DEST_MSG_INFO_BUF_FLOW_CTRL             (MSG_INFO_BUF_FLOW_CTRL+MSG_INFO_BUF_FLOW_CTRL_WIDTH)
#define       DEST_MSG_INFO_BUF_FLOW_CTRL_WIDTH         1
// set if REMOTE_SOURCE==1 and has mulicast enabled (i.e. this stream is part of a multicast group)
#define       REMOTE_SRC_IS_MCAST                     (DEST_MSG_INFO_BUF_FLOW_CTRL+DEST_MSG_INFO_BUF_FLOW_CTRL_WIDTH)
#define       REMOTE_SRC_IS_MCAST_WIDTH                 1
// set if no need to flush outgoing remote data from previous phase
#define       NO_PREV_PHASE_OUTGOING_DATA_FLUSH       (REMOTE_SRC_IS_MCAST+REMOTE_SRC_IS_MCAST_WIDTH)
#define       NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH   1
// Set to one to enable full credit flushing on src side
#define       SRC_FULL_CREDIT_FLUSH_EN                (NO_PREV_PHASE_OUTGOING_DATA_FLUSH+NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH)
#define       SRC_FULL_CREDIT_FLUSH_EN_WIDTH            1
// Set to one to enable full credit flushing on dest side
#define       DST_FULL_CREDIT_FLUSH_EN                (SRC_FULL_CREDIT_FLUSH_EN+SRC_FULL_CREDIT_FLUSH_EN_WIDTH)
#define       DST_FULL_CREDIT_FLUSH_EN_WIDTH            1
// Set to one to enable infinite messages per phase, accompanied by a last tile header bit which will end the phase
#define       INFINITE_PHASE_EN                       (DST_FULL_CREDIT_FLUSH_EN+DST_FULL_CREDIT_FLUSH_EN_WIDTH)
#define       INFINITE_PHASE_EN_WIDTH                   1
// Enables out-of-order phase execution by providing an array of size num_tiles at the end of phase blob, with order in which each tile should be sent. Each array entry contains a 17-bit tile address and a 15-bit tile size.
#define       OOO_PHASE_EXECUTION_EN                  (INFINITE_PHASE_EN+INFINITE_PHASE_EN_WIDTH)
#define       OOO_PHASE_EXECUTION_EN_WIDTH              1

// Properties of the remote source stream (coorindates, stream ID, and this streams destination index).
// Dont-care unless REMOTE_SOURCE == 1.
#define   STREAM_REMOTE_SRC_REG_INDEX   4
#define       STREAM_REMOTE_SRC_X                   0
#define       STREAM_REMOTE_SRC_X_WIDTH               NOC_ID_WIDTH
#define       STREAM_REMOTE_SRC_Y                   (STREAM_REMOTE_SRC_X+STREAM_REMOTE_SRC_X_WIDTH)
#define       STREAM_REMOTE_SRC_Y_WIDTH               NOC_ID_WIDTH
#define       REMOTE_SRC_STREAM_ID                  (STREAM_REMOTE_SRC_Y+STREAM_REMOTE_SRC_Y_WIDTH)
#define       REMOTE_SRC_STREAM_ID_WIDTH              STREAM_ID_WIDTH
#define       STREAM_REMOTE_SRC_DEST_INDEX          (REMOTE_SRC_STREAM_ID+REMOTE_SRC_STREAM_ID_WIDTH)
#define       STREAM_REMOTE_SRC_DEST_INDEX_WIDTH      STREAM_ID_WIDTH
#define       DRAM_READS__TRANS_SIZE_WORDS_LO       (STREAM_REMOTE_SRC_Y+STREAM_REMOTE_SRC_Y_WIDTH)
#define       DRAM_READS__TRANS_SIZE_WORDS_LO_WIDTH   12

// Remote source phase (may be different from the destination stream phase.)
// We use 20-bit phase ID, so phase count doesnt wrap until 1M phases.
// Dont-care unless REMOTE_SOURCE == 1.
#define   STREAM_REMOTE_SRC_PHASE_REG_INDEX   5
#define       DRAM_READS__SCRATCH_1_PTR             0
#define       DRAM_READS__SCRATCH_1_PTR_WIDTH         19
#define       DRAM_READS__TRANS_SIZE_WORDS_HI       (DRAM_READS__SCRATCH_1_PTR+DRAM_READS__SCRATCH_1_PTR_WIDTH)
#define       DRAM_READS__TRANS_SIZE_WORDS_HI_WIDTH   1

// 4-bit wide register that determines the threshold at which a stream
// with remote source sends an update message to STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE.
// Dont-care unless REMOTE_SOURCE==1.
// Values:
//   value[3:0] == 0 => disable threshold. Acks send as soon as any data are cleared/forwarded.
//   value[3:0] >  0 => threshold calculated according to the following formula:
//         if (value[3])
//              threshold = buf_size - (buf_size >> value[2:0])
//         else
//              threshold = (buf_size >> value[2:0])
//
// This enables setting thresholds of buf_size/2, buf_size/4, buf_size/8, ... buf_size/256,
// as well as  3*buf_size/4, 7*buf_size/8, etc.
#define   STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX   6

// Properties of the remote destination stream (coorindates, stream ID).  Dont-care unless REMOTE_RECEIVER == 1.
// If destination is multicast, this register specifies the starting coordinates of the destination
// multicast group/rectangle. (The end coordinates are in STREAM_MCAST_DEST below.)
#define   STREAM_REMOTE_DEST_REG_INDEX   7
#define       STREAM_REMOTE_DEST_X               0
#define       STREAM_REMOTE_DEST_X_WIDTH           NOC_ID_WIDTH
#define       STREAM_REMOTE_DEST_Y               (STREAM_REMOTE_DEST_X+STREAM_REMOTE_DEST_X_WIDTH)
#define       STREAM_REMOTE_DEST_Y_WIDTH           NOC_ID_WIDTH
#define       STREAM_REMOTE_DEST_STREAM_ID       (STREAM_REMOTE_DEST_Y+STREAM_REMOTE_DEST_Y_WIDTH)
#define       STREAM_REMOTE_DEST_STREAM_ID_WIDTH   STREAM_ID_WIDTH

// Properties of the local destination gather stream connection.
// Dont-care unless LOCAL_RECEIVER == 1.
// Shares register space with STREAM_REMOTE_DEST_REG_INDEX.
#define   STREAM_LOCAL_DEST_REG_INDEX   7
#define       STREAM_LOCAL_DEST_MSG_CLEAR_NUM       0
#define       STREAM_LOCAL_DEST_MSG_CLEAR_NUM_WIDTH   12
#define       STREAM_LOCAL_DEST_STREAM_ID           (STREAM_LOCAL_DEST_MSG_CLEAR_NUM+STREAM_LOCAL_DEST_MSG_CLEAR_NUM_WIDTH)
#define       STREAM_LOCAL_DEST_STREAM_ID_WIDTH       STREAM_ID_WIDTH

// Start address (in words) of the remote destination stream memory buffer.
#define   STREAM_REMOTE_DEST_BUF_START_REG_INDEX   8
#define       DRAM_WRITES__SCRATCH_1_PTR_LO       0
#define       DRAM_WRITES__SCRATCH_1_PTR_LO_WIDTH   16

// High bits for STREAM_REMOTE_DEST_BUF_START
#define   STREAM_REMOTE_DEST_BUF_START_HI_REG_INDEX   9

// Size (in words) of the remote destination stream memory buffer.
#define   STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX   10
#define       REMOTE_DEST_BUF_SIZE_WORDS          0
#define       REMOTE_DEST_BUF_SIZE_WORDS_WIDTH      MEM_WORD_ADDR_WIDTH
#define       DRAM_WRITES__SCRATCH_1_PTR_HI       0
#define       DRAM_WRITES__SCRATCH_1_PTR_HI_WIDTH   3

// Write pointer for the remote destination stream memory buffer.
// Can be written directly; automatically reset to 0 when
// STREAM_REMOTE_DEST_BUF_START is written.
#define   STREAM_REMOTE_DEST_WR_PTR_REG_INDEX   11

// Size (in power2) of the remote destination stream memory buffer.
// Bits encode powers of 2 sizes in words (2^(x+1)), e.g. 0 -> 2 words, 1 -> 4 words, 7 -> 256 words
// Max 256 word size.
// Only used when DEST_MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_REMOTE_DEST_MSG_INFO_BUF_SIZE_REG_INDEX   12
#define       REMOTE_DEST_MSG_INFO_BUF_SIZE_POW2       0
#define       REMOTE_DEST_MSG_INFO_BUF_SIZE_POW2_WIDTH   MSG_INFO_BUF_SIZE_POW_BITS

// Start address (in words) of the remote destination stream memory buffer.
// Only used when DEST_MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_REMOTE_DEST_MSG_INFO_BUF_START_REG_INDEX   13

// Write pointer for the remote destination message info buffer.
// Dont-care unless REMOTE_RECEIVER==1.
// Needs to be initialized to the start of the message info buffer of the remote destination
// at phase start, if destination is changed.
// Subsequently its incremented automatically as messages are forwarded.
// When DEST_MSG_INFO_BUF_FLOW_CTRL is true this pointer is the one above
#define   STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX   13

#ifdef RISC_B0_HW
// On WH B0, this register aliases STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX.
// It can be used to clear multiple tiles at once, even if they haven't been loaded into the msg info
// buffer. (Which is required for using STREAM_MSG_INFO/DATA_CLEAR_REG_INDEX.)  This way, we
// can clear however many pending messages have actually been received in L1.
// To clear N messages, we need to:
//   - Poll to ensure that the register value is 0 (indicating that previous clearing is done).
//   - Write (-2*N) in 2's complement (i.e. ~N+1) into the register.
// This performs both info and data clear steps at once.
// The register should be used only in RECEIVER_ENDPOINT mode.
#define   STREAM_RECEIVER_ENDPOINT_MULTI_TILE_CLEAR_REG_INDEX 13
#endif

// High bits for STREAM_REMOTE_DEST_MSG_INFO_BUF_START
// Only used when DEST_MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_REMOTE_DEST_MSG_INFO_BUF_START_HI_REG_INDEX   14

// High bits for STREAM_REMOTE_DEST_MSG_INFO_WR_PTR
// When DEST_MSG_INFO_BUF_FLOW_CTRL is true this pointer is the one above
#define   STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI_REG_INDEX   14

// Only used when DEST_MSG_INFO_BUF_FLOW_CTRL is true
// Write pointer for the remote destination message info buffer.
// Dont-care unless REMOTE_RECEIVER==1.
// Subsequently its incremented automatically as messages are forwarded.
#define   STREAM_REMOTE_DEST_MSG_INFO_WRAP_WR_PTR_REG_INDEX   15

// Priority for traffic sent to remote destination.
// Valid only for streams capable of remote sending.
// 4-bit value.
// Set to 0 to send traffic under round-robin arbitration.
// Set to 1-15 for priority arbitration (higher values are higher priority).
#define   STREAM_REMOTE_DEST_TRAFFIC_REG_INDEX   16
#define       NOC_PRIORITY         0
#define       NOC_PRIORITY_WIDTH     4
// set to one of the values (0-5) to select which VC unicast requests will be sent on
#define       UNICAST_VC_REG       (NOC_PRIORITY+NOC_PRIORITY_WIDTH)
#define       UNICAST_VC_REG_WIDTH   3

// Start address (in words) of the memory buffer associated with this stream.
#define   STREAM_BUF_START_REG_INDEX   17

// Stream buffer size (in words).
#define   STREAM_BUF_SIZE_REG_INDEX   18

// Read pointer value (word offset relative to buffer start).
// Can be updated by writing the register.
// Value does not guarantee that all data up to the current value have been sent
// off (forwarding command may be  ongoing).  To find out free space in the buffer,
// read STREAM_BUF_SPACE_AVAILABLE.
// Automatically reset to 0 when STREAM_BUF_START_REG is updated.
#define   STREAM_RD_PTR_REG_INDEX   19
#define       STREAM_RD_PTR_VAL        0
#define       STREAM_RD_PTR_VAL_WIDTH    MEM_WORD_ADDR_WIDTH
#define       STREAM_RD_PTR_WRAP       (STREAM_RD_PTR_VAL+STREAM_RD_PTR_VAL_WIDTH)
#define       STREAM_RD_PTR_WRAP_WIDTH   1

// Write pointer value (word offset relative to buffer start).
// Can be read to determine the location at which to write new data.
// Can be updated by writing the register.
// In normal operation, should be updated only by writing
// STREAM_NUM_MSGS_RECEIVED_INC_REG or STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG.
#define   STREAM_WR_PTR_REG_INDEX   20
#define       STREAM_WR_PTR_VAL        0
#define       STREAM_WR_PTR_VAL_WIDTH    MEM_WORD_ADDR_WIDTH
#define       STREAM_WR_PTR_WRAP       (STREAM_WR_PTR_VAL+STREAM_WR_PTR_VAL_WIDTH)
#define       STREAM_WR_PTR_WRAP_WIDTH   1

// Size (in power2) of the remote destination stream memory buffer.
// Bits encode powers of 2 sizes in words (2^(x+1)), e.g. 0 -> 2 words, 1 -> 4 words, 7 -> 256 words
// Max 256 word size.
// Only used when MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_MSG_INFO_BUF_SIZE_REG_INDEX   21
#define       MSG_INFO_BUF_SIZE_POW2       0
#define       MSG_INFO_BUF_SIZE_POW2_WIDTH   MSG_INFO_BUF_SIZE_POW_BITS

// Start address (in words) of the msg info buffer.
// Only used when MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_MSG_INFO_BUF_START_REG_INDEX   22

// Stream message info buffer address.
//
// This register needs to be initialized to the start of the message info buffer during
// phase configuration.  Subsequently it will be incremented by hardware as data are read
// from the buffer, thus doubling as the read pointer during phase execution.
//
// Stream hardware will assume that this buffer is large enough to hold info for all messages
// within a phase, so unlike the buffer, it never needs to wrap.
//
// The buffer is filled automatically by snooping for streams with remote source.
// For source enpoints, the buffer is written explicitly (along with the data buffer), after which
// STREAM_NUM_MSGS_RECEIVED_INC is written to notify the stream that messages are available for
// sending.
//
// Write pointer is also managed automatically by hardware, but can be read or reset using
// STREAM_MSG_INFO_WR_PTR_REG. Write pointer is also reset when writing this register.
// When MSG_INFO_BUF_FLOW_CTRL is true this pointer is the one above
#define   STREAM_MSG_INFO_PTR_REG_INDEX   22

// The read and write pointers for the msg info buffer when the message info buffer is in wrapping mode.
// Only used when MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_MSG_INFO_WRAP_RD_WR_PTR_REG_INDEX   23
#define       STREAM_MSG_INFO_WRAP_RD_PTR            0
#define       STREAM_MSG_INFO_WRAP_RD_PTR_WIDTH        MSG_INFO_BUF_SIZE_BITS
#define       STREAM_MSG_INFO_WRAP_RD_PTR_WRAP       (STREAM_MSG_INFO_WRAP_RD_PTR+STREAM_MSG_INFO_WRAP_RD_PTR_WIDTH)
#define       STREAM_MSG_INFO_WRAP_RD_PTR_WRAP_WIDTH   1
#define       STREAM_MSG_INFO_WRAP_WR_PTR            (STREAM_MSG_INFO_WRAP_RD_PTR_WRAP+STREAM_MSG_INFO_WRAP_RD_PTR_WRAP_WIDTH)
#define       STREAM_MSG_INFO_WRAP_WR_PTR_WIDTH        MSG_INFO_BUF_SIZE_BITS
#define       STREAM_MSG_INFO_WRAP_WR_PTR_WRAP       (STREAM_MSG_INFO_WRAP_WR_PTR+STREAM_MSG_INFO_WRAP_WR_PTR_WIDTH)
#define       STREAM_MSG_INFO_WRAP_WR_PTR_WRAP_WIDTH   1

// Write pointer value for message info buffer (absolute word address).
// In normal operation, should be updated only by writing
// STREAM_NUM_MSGS_RECEIVED_INC_REG or STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG.
// When MSG_INFO_BUF_FLOW_CTRL is true this pointer is the one above
#define   STREAM_MSG_INFO_WR_PTR_REG_INDEX   23

// Destination spec for multicasting streams. STREAM_MCAST_END_X/Y are
// the end coordinate for the multicast rectangle, with the ones from
// STREAM_REMOTE_DEST taken as start.
// Dont-care if STREAM_MCAST_EN == 0.
#define   STREAM_MCAST_DEST_REG_INDEX   24
#define       STREAM_MCAST_END_X                          0
#define       STREAM_MCAST_END_X_WIDTH                      NOC_ID_WIDTH
#define       STREAM_MCAST_END_Y                          (STREAM_MCAST_END_X+STREAM_MCAST_END_X_WIDTH)
#define       STREAM_MCAST_END_Y_WIDTH                      NOC_ID_WIDTH
#define       STREAM_MCAST_EN                             (STREAM_MCAST_END_Y+STREAM_MCAST_END_Y_WIDTH)
#define       STREAM_MCAST_EN_WIDTH                         1
#define       STREAM_MCAST_LINKED                         (STREAM_MCAST_EN+STREAM_MCAST_EN_WIDTH)
#define       STREAM_MCAST_LINKED_WIDTH                     1
// Set to 0 to select VC 4, and 1 to select VC 5 (default 0)
#define       STREAM_MCAST_VC                             (STREAM_MCAST_LINKED+STREAM_MCAST_LINKED_WIDTH)
#define       STREAM_MCAST_VC_WIDTH                         1
#define       STREAM_MCAST_NO_PATH_RES                    (STREAM_MCAST_VC+STREAM_MCAST_VC_WIDTH)
#define       STREAM_MCAST_NO_PATH_RES_WIDTH                1
#define       STREAM_MCAST_XY                             (STREAM_MCAST_NO_PATH_RES+STREAM_MCAST_NO_PATH_RES_WIDTH)
#define       STREAM_MCAST_XY_WIDTH                         1
#define       STREAM_MCAST_SRC_SIDE_DYNAMIC_LINKED        (STREAM_MCAST_XY+STREAM_MCAST_XY_WIDTH)
#define       STREAM_MCAST_SRC_SIDE_DYNAMIC_LINKED_WIDTH    1
#define       STREAM_MCAST_DEST_SIDE_DYNAMIC_LINKED       (STREAM_MCAST_SRC_SIDE_DYNAMIC_LINKED+STREAM_MCAST_SRC_SIDE_DYNAMIC_LINKED_WIDTH)
#define       STREAM_MCAST_DEST_SIDE_DYNAMIC_LINKED_WIDTH   1

// Number of multicast destinations (dont-care for non-multicast streams)
#define   STREAM_MCAST_DEST_NUM_REG_INDEX   25

// Specifies MSG_ARB_GROUP_SIZE. Valid values are 1 (round-robin
// arbitration between each incoming stream) or 4 (round-robin arbitration
// between groups of 4 incoming streams).
// Msg_LOCAL_STREAM_CLEAR_NUM specifies the number of messages that should
// be cleared from a gather stream before moving onto the next stream.
// When MSG_ARB_GROUP_SIZE > 1, the order of clearing the streams can be selected
// with MSG_GROUP_STREAM_CLEAR_TYPE. 0 = clear the whole group MSG_LOCAL_STREAM_CLEAR_NUM times,
// 1 = clear each stream of the group MSG_LOCAL_STREAM_CLEAR_NUM times before
// moving onto the next stream in the group.
#define   STREAM_GATHER_REG_INDEX   26
#define       MSG_LOCAL_STREAM_CLEAR_NUM           0
#define       MSG_LOCAL_STREAM_CLEAR_NUM_WIDTH       12
#define       MSG_GROUP_STREAM_CLEAR_TYPE          (MSG_LOCAL_STREAM_CLEAR_NUM+MSG_LOCAL_STREAM_CLEAR_NUM_WIDTH)
#define       MSG_GROUP_STREAM_CLEAR_TYPE_WIDTH      1
#define       MSG_ARB_GROUP_SIZE                   (MSG_GROUP_STREAM_CLEAR_TYPE+MSG_GROUP_STREAM_CLEAR_TYPE_WIDTH)
#define       MSG_ARB_GROUP_SIZE_WIDTH               3
#define       MSG_SRC_IN_ORDER_FWD                 (MSG_ARB_GROUP_SIZE+MSG_ARB_GROUP_SIZE_WIDTH)
#define       MSG_SRC_IN_ORDER_FWD_WIDTH             1
#define       MSG_SRC_ARBITRARY_CLEAR_NUM_EN       (MSG_SRC_IN_ORDER_FWD+MSG_SRC_IN_ORDER_FWD_WIDTH)
#define       MSG_SRC_ARBITRARY_CLEAR_NUM_EN_WIDTH   1

// When using in-order message forwarding, number of messages after which the source
// pointer goes back to zero (without phase change).
// Dont-care if STREAM_MCAST_EN == 0 or MSG_SRC_IN_ORDER_FWD == 0.
#define   STREAM_MSG_SRC_IN_ORDER_FWD_NUM_MSGS_REG_INDEX   27

// Actual phase number executed is STREAM_CURR_PHASE_BASE_REG_INDEX + STREAM_CURR_PHASE_REG_INDEX
// When reprogramming this register you must also reprogram STREAM_CURR_PHASE_REG_INDEX and STREAM_REMOTE_SRC_PHASE_REG_INDEX
#define   STREAM_CURR_PHASE_BASE_REG_INDEX   28

// Current phase number executed by the stream.
#define   STREAM_CURR_PHASE_REG_INDEX   29

// Actual address accessed will be STREAM_PHASE_AUTO_CFG_PTR_BASE_REG_INDEX + STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX
// When reprogramming this register you must also reprogram STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX
#define   STREAM_PHASE_AUTO_CFG_PTR_BASE_REG_INDEX   30

// Pointer to the stream auto-config data. Initialized to the start of
// the auto-config structure at workload start, automatically updated
// subsequenty.
// Specified as byte address, needs to be multiple of 4B.
#define   STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX   31

// This register acts as indirection to execute a phase that already exists somewhere in the blob.
// It can be used to compress the blob when many phases need to be repeated.
// When this register is written with a signed offset, the blob at address (auto_cfg pointer + offset) will be loaded.
// The loaded blob must manually set its phase (using STREAM_CURR_PHASE) for this feature to work correctly.
// Furthermore the phase after the reload blob phase must also set its current phase manually.
#define   STREAM_RELOAD_PHASE_BLOB_REG_INDEX   32

// Offset & size of the size field in the message header. Only valid offsets are multiples of 8
// (i.e. byte-aligned).
#define   STREAM_MSG_HEADER_FORMAT_REG_INDEX   33
#define       MSG_HEADER_WORD_CNT_OFFSET                       0
#define       MSG_HEADER_WORD_CNT_OFFSET_WIDTH                   MEM_WORD_BIT_OFFSET_WIDTH
#define       MSG_HEADER_WORD_CNT_BITS                         (MSG_HEADER_WORD_CNT_OFFSET+MSG_HEADER_WORD_CNT_OFFSET_WIDTH)
#define       MSG_HEADER_WORD_CNT_BITS_WIDTH                     MEM_WORD_BIT_OFFSET_WIDTH
#define       MSG_HEADER_INFINITE_PHASE_LAST_TILE_OFFSET       (MSG_HEADER_WORD_CNT_BITS+MSG_HEADER_WORD_CNT_BITS_WIDTH)
#define       MSG_HEADER_INFINITE_PHASE_LAST_TILE_OFFSET_WIDTH   MEM_WORD_BIT_OFFSET_WIDTH

// Register corresponding to the auto-configuration header. Written by each auto-config access
// at phase start, can be also written by software for initial configuration or if auto-config
// is disabled.
// PHASE_NUM_INCR is phase number increment relative to the previous executed phase (or 0 right
// after reset). The increment happens after auto-config is done, and before the phase is executed.
// (Therefore reading  STREAM_CURR_PHASE_REG while auto-config is ongoing, or if it hasnt started
// yet, may return the old phase number.)
// This enables up to 2^12-1 phases to be skipped. If more phases need to be skipped, it is
// necessary to insert an intermediate phase with zero messages, whose only purpose is to provide
// an additional skip offset.
#define   STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX   34
#define       PHASE_NUM_INCR                      0
#define       PHASE_NUM_INCR_WIDTH                  12
#define       CURR_PHASE_NUM_MSGS                 (PHASE_NUM_INCR+PHASE_NUM_INCR_WIDTH)
#define       CURR_PHASE_NUM_MSGS_WIDTH             12
#define       NEXT_PHASE_NUM_CFG_REG_WRITES       (CURR_PHASE_NUM_MSGS+CURR_PHASE_NUM_MSGS_WIDTH)
#define       NEXT_PHASE_NUM_CFG_REG_WRITES_WIDTH   8

// Should be written only for stream 0, applies to all streams.
#define   STREAM_PERF_CONFIG_REG_INDEX   35
#define       CLOCK_GATING_EN              0
#define       CLOCK_GATING_EN_WIDTH          1
#define       CLOCK_GATING_HYST            (CLOCK_GATING_EN+CLOCK_GATING_EN_WIDTH)
#define       CLOCK_GATING_HYST_WIDTH        7
// PARTIAL_SEND_WORDS_THR contols the minimum number of 16-byte words of a tile to accumulate in a relay stream before sending it off to the destination.
// If the size of the tile is less than or equal to PARTIAL_SEND_WORDS_THR, then this feild is ignored.
// Default is 16 words
#define       PARTIAL_SEND_WORDS_THR       (CLOCK_GATING_HYST+CLOCK_GATING_HYST_WIDTH)
#define       PARTIAL_SEND_WORDS_THR_WIDTH   8

// Scratch registers
// Exists only in streams 0-3 and 8-11
// Data can be stored at [23:0] from STREAM_SCRATCH_REG_INDEX + 0 to STREAM_SCRATCH_REG_INDEX + 5
// Can be loaded through overlay blobs.
#define   STREAM_SCRATCH_REG_INDEX   36

#define   STREAM_SCRATCH_0_REG_INDEX   36
#define       NCRISC_TRANS_EN                       0
#define       NCRISC_TRANS_EN_WIDTH                   1
#define       NCRISC_TRANS_EN_IRQ_ON_BLOB_END       (NCRISC_TRANS_EN + NCRISC_TRANS_EN_WIDTH)
#define       NCRISC_TRANS_EN_IRQ_ON_BLOB_END_WIDTH   1
#define       NCRISC_CMD_ID                         (NCRISC_TRANS_EN_IRQ_ON_BLOB_END + NCRISC_TRANS_EN_IRQ_ON_BLOB_END_WIDTH)
#define       NCRISC_CMD_ID_WIDTH                     3
// Kept for compatibility with grayskull, but doesnt not exist anymore in wormhole
#define       NEXT_NRISC_PIC_INT_ON_PHASE           (NCRISC_CMD_ID + NCRISC_CMD_ID_WIDTH)
#define       NEXT_NRISC_PIC_INT_ON_PHASE_WIDTH       19

#define   STREAM_SCRATCH_1_REG_INDEX   37
#define       DRAM_FIFO_RD_PTR_WORDS_LO               0
#define       DRAM_FIFO_RD_PTR_WORDS_LO_WIDTH           24
#define       NCRISC_LOOP_COUNT                       0
#define       NCRISC_LOOP_COUNT_WIDTH                   24
#define       NCRISC_INIT_ENABLE_BLOB_DONE_IRQ        0
#define       NCRISC_INIT_ENABLE_BLOB_DONE_IRQ_WIDTH    1
#define       NCRISC_INIT_DISABLE_BLOB_DONE_IRQ       (NCRISC_INIT_ENABLE_BLOB_DONE_IRQ + NCRISC_INIT_ENABLE_BLOB_DONE_IRQ_WIDTH)
#define       NCRISC_INIT_DISABLE_BLOB_DONE_IRQ_WIDTH   1

#define   STREAM_SCRATCH_2_REG_INDEX   38
#define       DRAM_FIFO_RD_PTR_WORDS_HI       0
#define       DRAM_FIFO_RD_PTR_WORDS_HI_WIDTH   4
#define       DRAM_FIFO_WR_PTR_WORDS_LO       (DRAM_FIFO_RD_PTR_WORDS_HI + DRAM_FIFO_RD_PTR_WORDS_HI_WIDTH)
#define       DRAM_FIFO_WR_PTR_WORDS_LO_WIDTH   20
#define       NCRISC_TOTAL_LOOP_ITER          0
#define       NCRISC_TOTAL_LOOP_ITER_WIDTH      24

#define   STREAM_SCRATCH_3_REG_INDEX   39
#define       DRAM_FIFO_WR_PTR_WORDS_HI                 0
#define       DRAM_FIFO_WR_PTR_WORDS_HI_WIDTH             8
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_LO           (DRAM_FIFO_WR_PTR_WORDS_HI + DRAM_FIFO_WR_PTR_WORDS_HI_WIDTH)
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_LO_WIDTH       16
#define       NCRISC_LOOP_INCR                          0
#define       NCRISC_LOOP_INCR_WIDTH                      16
#define       NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES       (NCRISC_LOOP_INCR+NCRISC_LOOP_INCR_WIDTH)
#define       NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES_WIDTH   8

#define   STREAM_SCRATCH_4_REG_INDEX   40
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_HI       0
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_HI_WIDTH   12
#define       DRAM_FIFO_BASE_ADDR_WORDS_LO          (DRAM_FIFO_CAPACITY_PTR_WORDS_HI + DRAM_FIFO_CAPACITY_PTR_WORDS_HI_WIDTH)
#define       DRAM_FIFO_BASE_ADDR_WORDS_LO_WIDTH      12
#define       NCRISC_LOOP_BACK_AUTO_CFG_PTR         0
#define       NCRISC_LOOP_BACK_AUTO_CFG_PTR_WIDTH     24

#define   STREAM_SCRATCH_5_REG_INDEX   41
#define       DRAM_FIFO_BASE_ADDR_WORDS_HI             0
#define       DRAM_FIFO_BASE_ADDR_WORDS_HI_WIDTH         16
// Processes the read or write operation to completeion without processing other dram streams in the meantime
#define       DRAM_EN_BLOCKING                         (DRAM_FIFO_BASE_ADDR_WORDS_HI + DRAM_FIFO_BASE_ADDR_WORDS_HI_WIDTH)
#define       DRAM_EN_BLOCKING_WIDTH                     1
// Fifo structure in dram holds a dram pointer and size that is used as indirection to a tile in dram
#define       DRAM_DATA_STRUCTURE_IS_LUT               (DRAM_EN_BLOCKING + DRAM_EN_BLOCKING_WIDTH)
#define       DRAM_DATA_STRUCTURE_IS_LUT_WIDTH           1
// During a dram read, if its detected that the fifo is empty the ncrisc will reset the read pointer back to base
// Its expected that there is no host interaction
#define       DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY       (DRAM_DATA_STRUCTURE_IS_LUT + DRAM_DATA_STRUCTURE_IS_LUT_WIDTH)
#define       DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY_WIDTH   1
// During a dram write, if its detected that the fifo is full the ncrisc will reset the write pointer back to base. Old data will be overwritten.
// Its expected that there is no host interaction
#define       DRAM_RESET_WR_PTR_TO_BASE_ON_FULL        (DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY + DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY_WIDTH)
#define       DRAM_RESET_WR_PTR_TO_BASE_ON_FULL_WIDTH    1
// The internal ncrisc rd/wr pointers will not be updated at phase end
// Its expected that there is no host interaction
#define       DRAM_NO_PTR_UPDATE_ON_PHASE_END          (DRAM_RESET_WR_PTR_TO_BASE_ON_FULL + DRAM_RESET_WR_PTR_TO_BASE_ON_FULL_WIDTH)
#define       DRAM_NO_PTR_UPDATE_ON_PHASE_END_WIDTH      1
// Before ending the phase the ncrisc will wait until the host has emptied the write buffer and then reset the read and write pointers to base
// This can be used for hosts that do not want to track wrapping
// The host must be aware of this behaviour for this functionality to work
#define       DRAM_WR_BUFFER_FLUSH_AND_RST_PTRS        (DRAM_NO_PTR_UPDATE_ON_PHASE_END + DRAM_NO_PTR_UPDATE_ON_PHASE_END_WIDTH)
#define       DRAM_WR_BUFFER_FLUSH_AND_RST_PTRS_WIDTH    1
#define       NCRISC_LOOP_NEXT_PIC_INT_ON_PHASE        0
#define       NCRISC_LOOP_NEXT_PIC_INT_ON_PHASE_WIDTH    20

// Start address (in words) of the message blob buffer.
// Only used when out-of-order execution is enabled. Read value consists of this register + current message blob offset.
#define   STREAM_MSG_BLOB_BUF_START_REG_INDEX   206

// Global offset table write entry interface.
#define   STREAM_GLOBAL_OFFSET_TABLE_REG_INDEX   207
#define       GLOBAL_OFFSET_VAL                   0
#define       GLOBAL_OFFSET_VAL_WIDTH               MEM_WORD_ADDR_WIDTH
#define       GLOBAL_OFFSET_TABLE_INDEX_SEL       (GLOBAL_OFFSET_VAL+GLOBAL_OFFSET_VAL_WIDTH)
#define       GLOBAL_OFFSET_TABLE_INDEX_SEL_WIDTH   GLOBAL_OFFSET_TABLE_SIZE_WIDTH
#define       GLOBAL_OFFSET_TABLE_CLEAR           (GLOBAL_OFFSET_TABLE_INDEX_SEL+GLOBAL_OFFSET_TABLE_INDEX_SEL_WIDTH)
#define       GLOBAL_OFFSET_TABLE_CLEAR_WIDTH       1

// Scratch location for firmware usage
// Guarantees that no side-effects occur in Overlay hardware
// Does not map to any actual registers in streams
#define   FIRMWARE_SCRATCH_REG_INDEX   208

// Bit mask of connnected local source. Dont care if LOCAL_SOURCES_CONNECTED == 0.
// Mask segments [23:0], [47:24], and [63:48] are at indexes STREAM_LOCAL_SRC_MASK_REG_INDEX,
// STREAM_LOCAL_SRC_MASK_REG_INDEX+1, STREAM_LOCAL_SRC_MASK_REG_INDEX+2.
#define   STREAM_LOCAL_SRC_MASK_REG_INDEX   224

// Reserved for msg header fetch interface
#define   STREAM_MSG_HEADER_FETCH_REG_INDEX   254

// Reserved for legacy reasons. This range appears not to be used in rtl anymore.
#define   RESERVED1_REG_INDEX   255

// Only in receiver endpoint/dram streams
// A 32 bit scratch register
#define   STREAM_SCRATCH32_REG_INDEX   256

// Status info for the stream.
#define   STREAM_WAIT_STATUS_REG_INDEX   257
// Set when stream is in START state with auto-config disabled, or if auto-config is enabled
// but PHASE_AUTO_ADVANCE=0
#define       WAIT_SW_PHASE_ADVANCE_SIGNAL                    0
#define       WAIT_SW_PHASE_ADVANCE_SIGNAL_WIDTH                1
// Set when stream has configured the current phase, but waits data from the previous one to be flushed.
#define       WAIT_PREV_PHASE_DATA_FLUSH                      (WAIT_SW_PHASE_ADVANCE_SIGNAL+WAIT_SW_PHASE_ADVANCE_SIGNAL_WIDTH)
#define       WAIT_PREV_PHASE_DATA_FLUSH_WIDTH                  1
// Set when stream is in data forwarding state.
#define       MSG_FWD_ONGOING                                 (WAIT_PREV_PHASE_DATA_FLUSH+WAIT_PREV_PHASE_DATA_FLUSH_WIDTH)
#define       MSG_FWD_ONGOING_WIDTH                             1
#define       STREAM_CURR_STATE                               (MSG_FWD_ONGOING+MSG_FWD_ONGOING_WIDTH)
#define       STREAM_CURR_STATE_WIDTH                           4
#define       TOKEN_GOTTEN                                    (STREAM_CURR_STATE+STREAM_CURR_STATE_WIDTH)
#define       TOKEN_GOTTEN_WIDTH                                1
#define       INFINITE_PHASE_END_DETECTED                     (TOKEN_GOTTEN+TOKEN_GOTTEN_WIDTH)
#define       INFINITE_PHASE_END_DETECTED_WIDTH                 1
#define       INFINITE_PHASE_END_HEADER_BUFFER_DETECTED       (INFINITE_PHASE_END_DETECTED+INFINITE_PHASE_END_DETECTED_WIDTH)
#define       INFINITE_PHASE_END_HEADER_BUFFER_DETECTED_WIDTH   1

// Only in receiver endpoint streams (stream 4 and 5)
// Read-only. Tells you the number of tiles that have arrived in L1
#define   STREAM_NUM_MSGS_RECEIVED_IN_BUF_AND_MEM_REG_INDEX   258

// Number of received & stored messages (read-only).
// To get the total number of messages penidng in memory read
// STREAM_NUM_MSGS_RECEIVED_IN_BUF_AND_MEM_REG_INDEX
#define   STREAM_NUM_MSGS_RECEIVED_REG_INDEX   259

// Available buffer space at the stream (in 16B words).
// Source cant send data unless available space > 0.
#define   STREAM_BUF_SPACE_AVAILABLE_REG_INDEX   260

// Available msg info buffer space at the stream (in 16B words).
// Source cant send data unless available space > 0.
// Only valid when MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_MSG_INFO_BUF_SPACE_AVAILABLE_REG_INDEX   261

// Memory address (in words) of the next in line received message (read-only).
#define   STREAM_NEXT_RECEIVED_MSG_ADDR_REG_INDEX   262

// Size in words of the next in line received message (read-only).
#define   STREAM_NEXT_RECEIVED_MSG_SIZE_REG_INDEX   263

// Clear message info, move read pointer, and reclaim buffer space for one or more stored messages.
// This is a special case of STREAM_MSG_INFO_CLEAR/STREAM_MSG_DATA_CLEAR where we arent streaming data
// and instead we just want to clear a bunch of messages after we have used them.
// If you are using streaming it is better to use STREAM_MSG_INFO_CLEAR/STREAM_MSG_DATA_CLEAR instead.
// You should not use both STREAM_MSG_INFO_CLEAR/STREAM_MSG_DATA_CLEAR and STREAM_MULTI_MSG_CLEAR at the same time
// Must be used only for streams where RECEIVER_ENDPOINT == 1.
#define   STREAM_MULTI_MSG_CLEAR_REG_INDEX   264

// Clear message info for one or more stored messages.  Only valid values are 1, 2, or 4.
// No effect on the read pointer.
// Should be used only for streams where RECEIVER_ENDPOINT == 1.
#define   STREAM_MSG_INFO_CLEAR_REG_INDEX   265

// Move read pointer & reclaim buffer space for one or more stored messages.
// Sends flow control update to the source if REMOTE_SOURCE==1.
// Only valid values are 1, 2, or 4.
// Should be used only for streams where RECEIVER_ENDPOINT == 1, after
// STREAM_MSG_INFO_CLEAR_REG has been written with the same value.
#define   STREAM_MSG_DATA_CLEAR_REG_INDEX   266

// Write-only. Write 1 to advance to the next phase if PHASE_AUTO_ADVANCE == 0.
#define   STREAM_PHASE_ADVANCE_REG_INDEX   267

// Write phase number to indicate destination ready for the given phase.
// (This is done automatically by stream hardware when starting a phase with REMOTE_SOURCE=1.)
// The phase number is the one indicated by STREAM_REMOTE_SRC_PHASE_REG at destination.
// This register is mapped to the shared destination ready table, not a per-stream register.
// (Stream index is taken from the register address, and stored into the table along with the
// phase number.)
#define   STREAM_DEST_PHASE_READY_UPDATE_REG_INDEX   268
#define       PHASE_READY_DEST_NUM           0
#define       PHASE_READY_DEST_NUM_WIDTH       6
#define       PHASE_READY_NUM                (PHASE_READY_DEST_NUM+PHASE_READY_DEST_NUM_WIDTH)
#define       PHASE_READY_NUM_WIDTH            20
// set if this stream is part of multicast group (i.e. if REMOTE_SRC_IS_MCAST==1)
#define       PHASE_READY_MCAST              (PHASE_READY_NUM+PHASE_READY_NUM_WIDTH)
#define       PHASE_READY_MCAST_WIDTH          1
// set if the message is in response to 2-way handshake
#define       PHASE_READY_TWO_WAY_RESP       (PHASE_READY_MCAST+PHASE_READY_MCAST_WIDTH)
#define       PHASE_READY_TWO_WAY_RESP_WIDTH   1

// Source ready message register for two-way handshake (sent by source in
// case destination ready entry is not found in the table).
// If received by a stream that already sent its ready update, it prompts resending.
#define   STREAM_SRC_READY_UPDATE_REG_INDEX   269
#define       STREAM_REMOTE_RDY_SRC_X        0
#define       STREAM_REMOTE_RDY_SRC_X_WIDTH    NOC_ID_WIDTH
#define       STREAM_REMOTE_RDY_SRC_Y        (STREAM_REMOTE_RDY_SRC_X+STREAM_REMOTE_RDY_SRC_X_WIDTH)
#define       STREAM_REMOTE_RDY_SRC_Y_WIDTH    NOC_ID_WIDTH
#define       REMOTE_RDY_SRC_STREAM_ID       (STREAM_REMOTE_RDY_SRC_Y+STREAM_REMOTE_RDY_SRC_Y_WIDTH)
#define       REMOTE_RDY_SRC_STREAM_ID_WIDTH   STREAM_ID_WIDTH
#define       IS_TOKEN_UPDATE                (REMOTE_RDY_SRC_STREAM_ID+REMOTE_RDY_SRC_STREAM_ID_WIDTH)
#define       IS_TOKEN_UPDATE_WIDTH            1

// Update available buffer space at remote destination stream.
// this is rd_ptr increment issued when a message is forwarded
#define   STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX   270
#define       REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM       0
#define       REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH   6
#define       REMOTE_DEST_BUF_WORDS_FREE_INC                        (REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM+REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH)
#define       REMOTE_DEST_BUF_WORDS_FREE_INC_WIDTH                    MEM_WORD_ADDR_WIDTH
#define       REMOTE_DEST_MSG_INFO_BUF_WORDS_FREE_INC               (REMOTE_DEST_BUF_WORDS_FREE_INC+REMOTE_DEST_BUF_WORDS_FREE_INC_WIDTH)
#define       REMOTE_DEST_MSG_INFO_BUF_WORDS_FREE_INC_WIDTH           MSG_INFO_BUF_SIZE_WORDS_WIDTH

// Write to reset & stop stream.
#define   STREAM_RESET_REG_INDEX   271

// AND value of zero masks for the pending message group.
// (Header bits [95:64].)
// Read-only.  Valid only for receiver endpoint streams.
#define   STREAM_MSG_GROUP_ZERO_MASK_AND_REG_INDEX   272

// Returns 1 if the message info register is full (read-only).
#define   STREAM_MSG_INFO_FULL_REG_INDEX   273

// Returns 1 if the message info register is full (read-only), and there are no outstanding loads in progress.
#define   STREAM_MSG_INFO_FULLY_LOADED_REG_INDEX   274

// Returns 1 if the message info register can accept new message push (read-only).
// Equivalent to checking the condition:
//   (STREAM_MSG_INFO_FULL_REG_INDEX == 0) && (STREAM_MSG_INFO_PTR_REG_INDEX == STREAM_MSG_INFO_WR_PTR_REG_INDEX)
// (I.e. ther is free space in the msg info register, and we dont have any message info headers in the
//  memory buffer about to be fetched.)
#define   STREAM_MSG_INFO_CAN_PUSH_NEW_MSG_REG_INDEX   275

// Concat compress flags from 4 tiles in the pending message group.
// (Header bit 52.)
// Read-only.  Valid only for receiver endpoint streams.
#define   STREAM_MSG_GROUP_COMPRESS_REG_INDEX   276

// Returns 1 if all msgs that the phase can accept have been pushed into the stream. 0 otherwise.
#define   STREAM_PHASE_ALL_MSGS_PUSHED_REG_INDEX   277

// Returns 1 if the stream is in a state where it can accept msgs.
#define   STREAM_READY_FOR_MSG_PUSH_REG_INDEX   278

// Returns global offset table entry 0. The rest of the table entries can be read at index
// STREAM_GLOBAL_OFFSET_TABLE_RD_REG_INDEX+i, up to maximum entry size.
#define   STREAM_GLOBAL_OFFSET_TABLE_RD_REG_INDEX   279

// 32 bit register. Each bit denotes whether the corresponding stream has completed its blob run and is in idle state.
// Resets to 0 upon starting a new stream run. Initially all are 0 to exclude streams that might not be used.
// Can be manually reset to 0 by writing 1 to the corresponding bit.
// Exists only in stream 0
#define   STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX   288

// Reading this register will give you a stream id of a stream that finished its blob (according to STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX)
// Subsequent reads will give you the next stream, untill all streams are read, after which it will loop
// This register is only valid if BLOB_NEXT_AUTO_CFG_DONE_VALID is set (i.e. if STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX non-zero)
// Exists only in stream 0
#define   STREAM_BLOB_NEXT_AUTO_CFG_DONE_REG_INDEX   290
#define       BLOB_NEXT_AUTO_CFG_DONE_STREAM_ID       0
#define       BLOB_NEXT_AUTO_CFG_DONE_STREAM_ID_WIDTH   STREAM_ID_WIDTH
#define       BLOB_NEXT_AUTO_CFG_DONE_VALID           16
#define       BLOB_NEXT_AUTO_CFG_DONE_VALID_WIDTH       1

// For receiver endpoint streams that expose the full message header bus to unpacker,
// write this register to specify the full header in case the stream is not snooping
// a remote source but instead also works as a source endpoint.
// Write (STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX+i) to set bits [i*32 +: 32]
// of the message header for the next message, prior to writing STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX.
#define   STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX   291

// Available buffer space at remote destination stream(s) for both the data buffer and msg info buffer.
// Dont care unless REMOTE_RECEIVER == 1.
// Source cant send data unless WORDS_FREE > 0.
// Read-only; updated automatically to maximum value when
// STREAM_REMOTE_DEST_BUF_SIZE_REG/STREAM_REMOTE_DEST_MSG_INFO_BUF_SIZE_REG is updated.
// For multicast streams, values for successive destinations are at
// subsequent indexes (STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX+1,
// STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX+2, etc.).
// REMOTE_DEST_MSG_INFO_WORDS_FREE is only valid when DEST_MSG_INFO_BUF_FLOW_CTRL is true
#define   STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX   297
#define       REMOTE_DEST_WORDS_FREE                0
#define       REMOTE_DEST_WORDS_FREE_WIDTH            MEM_WORD_ADDR_WIDTH
#define       REMOTE_DEST_MSG_INFO_WORDS_FREE       (REMOTE_DEST_WORDS_FREE+REMOTE_DEST_WORDS_FREE_WIDTH)
#define       REMOTE_DEST_MSG_INFO_WORDS_FREE_WIDTH   MSG_INFO_BUF_SIZE_WORDS_WIDTH

// Read-only register view of the bits on the o_full_msg_info bus.
// Exposed as 32-bit read-only registers starting at this index.
#define   STREAM_RECEIVER_MSG_INFO_REG_INDEX   329

// Debug bus stream selection. Write the stream id for the stream that you want exposed on the debug bus
// This register only exists in stream 0.
#define   STREAM_DEBUG_STATUS_SEL_REG_INDEX   499
#define       DEBUG_STATUS_STREAM_ID_SEL        0
#define       DEBUG_STATUS_STREAM_ID_SEL_WIDTH    STREAM_ID_WIDTH
#define       DISABLE_DEST_READY_TABLE          (DEBUG_STATUS_STREAM_ID_SEL+DEBUG_STATUS_STREAM_ID_SEL_WIDTH)
#define       DISABLE_DEST_READY_TABLE_WIDTH      1
#define       DISABLE_GLOBAL_OFFSET_TABLE       (DISABLE_DEST_READY_TABLE+DISABLE_DEST_READY_TABLE_WIDTH)
#define       DISABLE_GLOBAL_OFFSET_TABLE_WIDTH   1

// Debugging: Non-zero value indicates an invalid stream operation occured.
// Sticky, write 1 to clear.
#define   STREAM_DEBUG_ASSERTIONS_REG_INDEX   500

// Read-only register that exposes internal states of the stream.
// Useful for debugging. Valid 32-bit data from STREAM_DEBUG_STATUS_REG_INDEX + 0 to STREAM_DEBUG_STATUS_REG_INDEX + 9
#define   STREAM_DEBUG_STATUS_REG_INDEX   501

// Reserved for legacy reasons. This range appears not to be used in rtl anymore.
#define   RESERVED2_REG_INDEX   511

#endif // def NOC_OVERLAY_PARAMETERS_H
