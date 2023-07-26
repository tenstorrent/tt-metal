
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AUTO_GENERATED! DO NOT MODIFY!                                                                              //
// Please run                                                                                                  //
//                                                                                                             //
// (echo '<% type=:c_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.h     //
// (echo '<% type=:cpp_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.hpp //
// Open noc_overlay_parameters.hpp and move static class varaible definitions to noc_overlay_parameters.cpp    //
// overriding existing ones.                                                                                   //
//                                                                                                             //
// to regenerate                                                                                               //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef NOC_OVERLAY_PARAMETERS_H
#define NOC_OVERLAY_PARAMETERS_H

#ifndef NOC_OVERLAY_PARAMETERS_BASIC_H
#define NOC_OVERLAY_PARAMETERS_BASIC_H

#define NOC_NUM_STREAMS 64

#define NUM_MCAST_STREAM_ID_START 0
#define NUM_MCAST_STREAM_ID_END   3
#define NUM_RECEIVER_ENDPOINT_STREAM_ID_START 4
#define NUM_RECEIVER_ENDPOINT_STREAM_ID_END   5
#define NUM_REMOTE_RECEIVER_STREAM_ID_START 0
#define NUM_REMOTE_RECEIVER_STREAM_ID_END 63
#define RECEIVER_ENDPOINT_STREAM_MSG_GROUP_SIZE 4
#define RECEIVER_ENDPOINT_STREAM_MSG_INFO_FIFO_GROUPS     2
#define NON_RECEIVER_ENDPOINT_STREAM_MSG_INFO_FIFO_GROUPS 2
#define DEST_READY_COMMON_CACHE_NUM_ENTRIES 24
#define DEST_READY_MCAST_CACHE_NUM_ENTRIES 8

#define NOC_OVERLAY_START_ADDR     0xFFB40000
#define NOC_STREAM_REG_SPACE_SIZE  0x1000

#define STREAM_REG_ADDR(stream_id, reg_id) ((NOC_OVERLAY_START_ADDR) + (((uint32_t)(stream_id))*(NOC_STREAM_REG_SPACE_SIZE)) + (((uint32_t)(reg_id)) << 2))

#define NOC0_REGS_START_ADDR       0xFFB20000
#define NOC1_REGS_START_ADDR       0xFFB30000

#define NCRISC_STREAM_RANGE_1_START 0
#define NCRISC_STREAM_RANGE_1_END   3
#define NCRISC_STREAM_RANGE_2_START 8
#define NCRISC_STREAM_RANGE_2_END   11
#define NCRISC_PIC_CONFIG_PHASE_DEFAULT           0


#define NOC_STREAM_WRITE_REG(stream_id, reg_id, val)  ((*((volatile tt_reg_ptr uint32_t*)(STREAM_REG_ADDR(stream_id, reg_id)))) = (val))
#define NOC_STREAM_READ_REG(stream_id, reg_id)        (*((volatile tt_reg_ptr uint32_t*)(STREAM_REG_ADDR(stream_id, reg_id))))

#define NOC_STREAM_WRITE_REG_FIELD(stream_id, reg_id, field, val) (NOC_STREAM_WRITE_REG(stream_id, reg_id, ((NOC_STREAM_READ_REG(stream_id, reg_id) & ~((1 << field##_WIDTH) - 1)) | ((val & ((1 << field##_WIDTH) - 1)) << field))))
#define NOC_STREAM_READ_REG_FIELD(stream_id, reg_id, field)       ((NOC_STREAM_READ_REG(stream_id, reg_id) >> field) & ((1 << field##_WIDTH) - 1))
#define NOC_STREAM_GET_REG_FIELD(reg_val, field)       (((reg_val) >> field) & ((1 << field##_WIDTH) - 1))

#define NOC_WRITE_REG(addr, val) ((*((volatile tt_reg_ptr uint32_t*)(addr)))) = (val)
#define NOC_READ_REG(addr)       (*((volatile tt_reg_ptr uint32_t*)(addr)))


#define NOC_ID_WIDTH     6
#define STREAM_ID_WIDTH  6

#define DEST_CNT_WIDTH   6
#define NOC_NUM_WIDTH     1

#define STREAM_REG_INDEX_WIDTH 8
#define STREAM_REG_CFG_DATA_WIDTH 24

#define MEM_WORD_ADDR_WIDTH 16

#define MEM_WORD_WIDTH 16

#define MEM_WORD_BIT_OFFSET_WIDTH 7

#endif


// TODO: verify that all registers are used and updated properly

// Properties of the remote source stream (coorindates, stream ID, and this streams destination index).
// Dont-care unless REMOTE_SOURCE == 1.
#define   STREAM_REMOTE_SRC_REG_INDEX   0
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
#define   STREAM_REMOTE_SRC_PHASE_REG_INDEX   1
#define       DRAM_READS__SCRATCH_1_PTR             0
#define       DRAM_READS__SCRATCH_1_PTR_WIDTH         18
#define       DRAM_READS__TRANS_SIZE_WORDS_HI       (DRAM_READS__SCRATCH_1_PTR+DRAM_READS__SCRATCH_1_PTR_WIDTH)
#define       DRAM_READS__TRANS_SIZE_WORDS_HI_WIDTH   4

// Properties of the remote destination stream (coorindates, stream ID).  Dont-care unless REMOTE_RECEIVER == 1.
// If destination is multicast, this register specifies the starting coordinates of the destination
// multicast group/rectangle. (The end coordinates are in STREAM_MCAST_DEST below.)
#define   STREAM_REMOTE_DEST_REG_INDEX   2
#define       STREAM_REMOTE_DEST_X               0
#define       STREAM_REMOTE_DEST_X_WIDTH           NOC_ID_WIDTH
#define       STREAM_REMOTE_DEST_Y               (STREAM_REMOTE_DEST_X+STREAM_REMOTE_DEST_X_WIDTH)
#define       STREAM_REMOTE_DEST_Y_WIDTH           NOC_ID_WIDTH
#define       STREAM_REMOTE_DEST_STREAM_ID       (STREAM_REMOTE_DEST_Y+STREAM_REMOTE_DEST_Y_WIDTH)
#define       STREAM_REMOTE_DEST_STREAM_ID_WIDTH   STREAM_ID_WIDTH

// Start address (in words) of the remote destination stream memory buffer.
#define   STREAM_REMOTE_DEST_BUF_START_REG_INDEX   3
#define       DRAM_WRITES__SCRATCH_1_PTR_LO       0
#define       DRAM_WRITES__SCRATCH_1_PTR_LO_WIDTH   16

// Size (in words) of the remote destination stream memory buffer.
#define   STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX   4
#define       REMOTE_DEST_BUF_SIZE_WORDS          0
#define       REMOTE_DEST_BUF_SIZE_WORDS_WIDTH      16
#define       DRAM_WRITES__SCRATCH_1_PTR_HI       0
#define       DRAM_WRITES__SCRATCH_1_PTR_HI_WIDTH   2

// Write pointer for the remote destination stream memory buffer.
// Can be written directly; automatically reset to 0 when
// STREAM_REMOTE_DEST_BUF_START is written.
#define   STREAM_REMOTE_DEST_WR_PTR_REG_INDEX   5

// Start address (in words) of the memory buffer associated with this stream.
#define   STREAM_BUF_START_REG_INDEX   6

// Stream buffer size (in words).
#define   STREAM_BUF_SIZE_REG_INDEX   7

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
#define   STREAM_MSG_INFO_PTR_REG_INDEX   8

// Write pointer for the remote destination message info buffer.
// Dont-care unless REMOTE_RECEIVER==1.
// Needs to be initialized to the start of the message info buffer of the remote destination
// at phase start, if destination is changed.
// Subsequently its incremented automatically as messages are forwarded.
#define   STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX   9

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
//   * Phase/data forward options:
//      PHASE_AUTO_CONFIG = set to 1 for stream to fetch next phase configuration automatically.
//      PHASE_AUTO_ADVANCE = set to 1 for stream to advance to next phase automatically
//            (otherwise need to write STREAM_PHASE_ADVANCE below)
//      DATA_AUTO_SEND = set to 1 to forward data automatically based on read/write pointers;
//             set to 0 to forward data only when STREAM_NEXT_MSG_SEND is written
#define   STREAM_MISC_CFG_REG_INDEX   10
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
#define       PHASE_AUTO_CONFIG                       (REMOTE_RECEIVER+REMOTE_RECEIVER_WIDTH)
#define       PHASE_AUTO_CONFIG_WIDTH                   1
#define       PHASE_AUTO_ADVANCE                      (PHASE_AUTO_CONFIG+PHASE_AUTO_CONFIG_WIDTH)
#define       PHASE_AUTO_ADVANCE_WIDTH                  1
#define       DATA_AUTO_SEND                          (PHASE_AUTO_ADVANCE+PHASE_AUTO_ADVANCE_WIDTH)
#define       DATA_AUTO_SEND_WIDTH                      1
#define       NEXT_PHASE_SRC_CHANGE                   (DATA_AUTO_SEND+DATA_AUTO_SEND_WIDTH)
#define       NEXT_PHASE_SRC_CHANGE_WIDTH               1
#define       NEXT_PHASE_DEST_CHANGE                  (NEXT_PHASE_SRC_CHANGE+NEXT_PHASE_SRC_CHANGE_WIDTH)
#define       NEXT_PHASE_DEST_CHANGE_WIDTH              1
// set if REMOTE_SOURCE==1 and the buffer is large enough to accept full phase data without wrapping:
#define       DATA_BUF_NO_FLOW_CTRL                   (NEXT_PHASE_DEST_CHANGE+NEXT_PHASE_DEST_CHANGE_WIDTH)
#define       DATA_BUF_NO_FLOW_CTRL_WIDTH               1
// set if REMOTE_RECEIVER==1 and the destination buffer is large enough to accept full phase data without wrapping:
#define       DEST_DATA_BUF_NO_FLOW_CTRL              (DATA_BUF_NO_FLOW_CTRL+DATA_BUF_NO_FLOW_CTRL_WIDTH)
#define       DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH          1
// set if REMOTE_SOURCE==1 and has mulicast enabled (i.e. this stream is part of a multicast group)
#define       REMOTE_SRC_IS_MCAST                     (DEST_DATA_BUF_NO_FLOW_CTRL+DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH)
#define       REMOTE_SRC_IS_MCAST_WIDTH                 1
// set if no need to flush outgoing remote data from previous phase
#define       NO_PREV_PHASE_OUTGOING_DATA_FLUSH       (REMOTE_SRC_IS_MCAST+REMOTE_SRC_IS_MCAST_WIDTH)
#define       NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH   1
// set to one of the values (0-5) to select which VC unicast requests will be sent on
#define       UNICAST_VC_REG                          (NO_PREV_PHASE_OUTGOING_DATA_FLUSH+NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH)
#define       UNICAST_VC_REG_WIDTH                      3
// set to one of the values (0-5) to select which VC control flow updates will be sent on
#define       REG_UPDATE_VC_REG                       (UNICAST_VC_REG+UNICAST_VC_REG_WIDTH)
#define       REG_UPDATE_VC_REG_WIDTH                   3

// Current phase number executed by the stream.
#define   STREAM_CURR_PHASE_REG_INDEX   11

// Pointer to the stream auto-config data. Initialized to the start of
// the auto-config structure at workload start, automatically updated
// subsequenty.
// Specified as byte address, needs to be multiple of 4B.
#define   STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX   12

// Destination spec for multicasting streams. STREAM_MCAST_END_X/Y are
// the end coordinate for the multicast rectangle, with the ones from
// STREAM_REMOTE_DEST taken as start.
// Dont-care if STREAM_MCAST_EN == 0.
//
// Also specifies MSG_ARB_GROUP_SIZE; valid values are 1 (round-robin
// arbitration between each incoming stream) or 4 (round-robin arbitration
// between groups of 4 incoming streams).
#define   STREAM_MCAST_DEST_REG_INDEX   13
#define       STREAM_MCAST_END_X             0
#define       STREAM_MCAST_END_X_WIDTH         NOC_ID_WIDTH
#define       STREAM_MCAST_END_Y             (STREAM_MCAST_END_X+STREAM_MCAST_END_X_WIDTH)
#define       STREAM_MCAST_END_Y_WIDTH         NOC_ID_WIDTH
#define       STREAM_MCAST_EN                (STREAM_MCAST_END_Y+STREAM_MCAST_END_Y_WIDTH)
#define       STREAM_MCAST_EN_WIDTH            1
#define       STREAM_MCAST_LINKED            (STREAM_MCAST_EN+STREAM_MCAST_EN_WIDTH)
#define       STREAM_MCAST_LINKED_WIDTH        1
// Set to 0 to select VC 4, and 1 to select VC 5 (default 0)
#define       STREAM_MCAST_VC                (STREAM_MCAST_LINKED+STREAM_MCAST_LINKED_WIDTH)
#define       STREAM_MCAST_VC_WIDTH            1
#define       STREAM_MCAST_NO_PATH_RES       (STREAM_MCAST_VC+STREAM_MCAST_VC_WIDTH)
#define       STREAM_MCAST_NO_PATH_RES_WIDTH   1
#define       MSG_ARB_GROUP_SIZE             (STREAM_MCAST_NO_PATH_RES+STREAM_MCAST_NO_PATH_RES_WIDTH)
#define       MSG_ARB_GROUP_SIZE_WIDTH         3
#define       MSG_SRC_IN_ORDER_FWD           (MSG_ARB_GROUP_SIZE+MSG_ARB_GROUP_SIZE_WIDTH)
#define       MSG_SRC_IN_ORDER_FWD_WIDTH       1
#define       STREAM_MCAST_XY                (MSG_SRC_IN_ORDER_FWD+MSG_SRC_IN_ORDER_FWD_WIDTH)
#define       STREAM_MCAST_XY_WIDTH            1

// When using in-order message forwarding, number of messages after which the source
// pointer goes back to zero (without phase change).
// Dont-care if STREAM_MCAST_EN == 0 or MSG_SRC_IN_ORDER_FWD == 0.
#define   STREAM_MSG_SRC_IN_ORDER_FWD_NUM_MSGS_REG_INDEX   14

// Number of multicast destinations (dont-care for non-multicast streams)
#define   STREAM_MCAST_DEST_NUM_REG_INDEX   15

// Offset & size of the size field in the message header. Only valid offsets are multiples of 8
// (i.e. byte-aligned).
#define   STREAM_MSG_HEADER_FORMAT_REG_INDEX   16
#define       MSG_HEADER_WORD_CNT_OFFSET       0
#define       MSG_HEADER_WORD_CNT_OFFSET_WIDTH   MEM_WORD_BIT_OFFSET_WIDTH
#define       MSG_HEADER_WORD_CNT_BITS         (MSG_HEADER_WORD_CNT_OFFSET+MSG_HEADER_WORD_CNT_OFFSET_WIDTH)
#define       MSG_HEADER_WORD_CNT_BITS_WIDTH     MEM_WORD_BIT_OFFSET_WIDTH

// Number of received & stored messages (read-only).
#define   STREAM_NUM_MSGS_RECEIVED_REG_INDEX   17

// Memory address (in words) of the next in line received message (read-only).
#define   STREAM_NEXT_RECEIVED_MSG_ADDR_REG_INDEX   18

// Size in words of the next in line received message (read-only).
#define   STREAM_NEXT_RECEIVED_MSG_SIZE_REG_INDEX   19

// Clear message info for one or more stored messages.  Only valid values are 1, 2, or 4.
// No effect on the read pointer.
// Should be used only for streams where RECEIVER_ENDPOINT == 1.
#define   STREAM_MSG_INFO_CLEAR_REG_INDEX   20

// Move read pointer & reclaim buffer space for one or more stored messages.
// Sends flow control update to the source if REMOTE_SOURCE==1.
// Only valid values are 1, 2, or 4.
// Should be used only for streams where RECEIVER_ENDPOINT == 1, after
// STREAM_MSG_INFO_CLEAR_REG has been written with the same value.
#define   STREAM_MSG_DATA_CLEAR_REG_INDEX   21

// Write to send the next in line stored message. Used when DATA_AUTO_SEND == 0.
#define   STREAM_NEXT_MSG_SEND_REG_INDEX   22

// Read pointer value (word offset relative to buffer start). Can be updated by
// writing the register (e.g. to force resend).
// Value does not guarantee that all data up to the current value have been sent
// off (forwarding command may be  ongoing).  To find out free space in the buffer,
// read STREAM_BUF_SPACE_AVAILABLE.
// Automatically reset to 0 when STREAM_BUF_START_REG is updated.
#define   STREAM_RD_PTR_REG_INDEX   23

// Write pointer value (word offset relative to buffer start).
// Can be read to determine the location at which to write new data.
// In normal operation, should be updated only by writing
// STREAM_NUM_MSGS_RECEIVED_INC_REG or STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG.
#define   STREAM_WR_PTR_REG_INDEX   24

// Write pointer value for message info buffer (absolute word address).
// In normal operation, should be updated only by writing
// STREAM_NUM_MSGS_RECEIVED_INC_REG or STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG.
#define   STREAM_MSG_INFO_WR_PTR_REG_INDEX   25

// Write-only. Write 1 to advance to the next phase if PHASE_AUTO_ADVANCE == 0.
#define   STREAM_PHASE_ADVANCE_REG_INDEX   26

// Available buffer space at the stream (in 16B words).
// Source cant send data unless available space > 0.
#define   STREAM_BUF_SPACE_AVAILABLE_REG_INDEX   27

// For endpoints with SOURCE_ENDPOINT == 1, this register is for firmware
// to register new message for sending.
// This updates the msg_info register structure directly, rather than writing to the message info
// buffer in memory.
// Must not be written when the message info register structure is full, or if
// there are message info entries in the memory buffer. (This would cause a race
// condition.)
#define   STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX   28
#define       SOURCE_ENDPOINT_NEW_MSG_ADDR       0
#define       SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH   MEM_WORD_ADDR_WIDTH
#define       SOURCE_ENDPOINT_NEW_MSG_SIZE       (SOURCE_ENDPOINT_NEW_MSG_ADDR+SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH)
#define       SOURCE_ENDPOINT_NEW_MSG_SIZE_WIDTH   MEM_WORD_ADDR_WIDTH

// For endpoints with SOURCE_ENDPOINT == 1, this register is for firmware
// to update the number of messages whose data & header are available in the memory buffer.
// Hardware register is incremented atomically if sending of previous messages is in progress.
#define   STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX   29
#define       SOURCE_ENDPOINT_NEW_MSGS_NUM              0
#define       SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH          12
#define       SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE       (SOURCE_ENDPOINT_NEW_MSGS_NUM+SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH)
#define       SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE_WIDTH   MEM_WORD_ADDR_WIDTH

// Write to reset & stop stream.
#define   STREAM_RESET_REG_INDEX   30

// Write phase number to indicate destination ready for the given phase.
// (This is done automatically by stream hardware when starting a phase with REMOTE_SOURCE=1.)
// The phase number is the one indicated by STREAM_REMOTE_SRC_PHASE_REG at destination.
// This register is mapped to the shared destination ready table, not a per-stream register.
// (Stream index is taken from the register address, and stored into the table along with the
// phase number.)
#define   STREAM_DEST_PHASE_READY_UPDATE_REG_INDEX   31
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
#define   STREAM_SRC_READY_UPDATE_REG_INDEX   32
#define       STREAM_REMOTE_RDY_SRC_X        0
#define       STREAM_REMOTE_RDY_SRC_X_WIDTH    NOC_ID_WIDTH
#define       STREAM_REMOTE_RDY_SRC_Y        (STREAM_REMOTE_RDY_SRC_X+STREAM_REMOTE_RDY_SRC_X_WIDTH)
#define       STREAM_REMOTE_RDY_SRC_Y_WIDTH    NOC_ID_WIDTH
#define       REMOTE_RDY_SRC_STREAM_ID       (STREAM_REMOTE_RDY_SRC_Y+STREAM_REMOTE_RDY_SRC_Y_WIDTH)
#define       REMOTE_RDY_SRC_STREAM_ID_WIDTH   STREAM_ID_WIDTH

// Update available buffer space at remote destination stream.
// this is rd_ptr increment issued when a message is forwarded
#define   STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX   33
#define       REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM       0
#define       REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH   6
#define       REMOTE_DEST_BUF_WORDS_FREE_INC                        (REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM+REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH)
#define       REMOTE_DEST_BUF_WORDS_FREE_INC_WIDTH                    16

// Status info for the stream.
#define   STREAM_WAIT_STATUS_REG_INDEX   34
// Set when stream is in START state with auto-config disabled, or if auto-config is enabled
// but PHASE_AUTO_ADVANCE=0
#define       WAIT_SW_PHASE_ADVANCE_SIGNAL       0
#define       WAIT_SW_PHASE_ADVANCE_SIGNAL_WIDTH   1
// Set when stream has configured the current phase, but waits data from the previous one to be flushed.
#define       WAIT_PREV_PHASE_DATA_FLUSH         (WAIT_SW_PHASE_ADVANCE_SIGNAL+WAIT_SW_PHASE_ADVANCE_SIGNAL_WIDTH)
#define       WAIT_PREV_PHASE_DATA_FLUSH_WIDTH     1
// Set when stream is in data forwarding state.
#define       MSG_FWD_ONGOING                    (WAIT_PREV_PHASE_DATA_FLUSH+WAIT_PREV_PHASE_DATA_FLUSH_WIDTH)
#define       MSG_FWD_ONGOING_WIDTH                1
#define       STREAM_CURR_STATE                  (MSG_FWD_ONGOING+MSG_FWD_ONGOING_WIDTH)
#define       STREAM_CURR_STATE_WIDTH              4

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
#define   STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX   35
#define       PHASE_NUM_INCR                      0
#define       PHASE_NUM_INCR_WIDTH                  12
#define       CURR_PHASE_NUM_MSGS                 (PHASE_NUM_INCR+PHASE_NUM_INCR_WIDTH)
#define       CURR_PHASE_NUM_MSGS_WIDTH             12
#define       NEXT_PHASE_NUM_CFG_REG_WRITES       (CURR_PHASE_NUM_MSGS+CURR_PHASE_NUM_MSGS_WIDTH)
#define       NEXT_PHASE_NUM_CFG_REG_WRITES_WIDTH   8

// Should be written only for stream 0, applies to all streams.
#define   STREAM_PERF_CONFIG_REG_INDEX   36
#define       CLOCK_GATING_EN              0
#define       CLOCK_GATING_EN_WIDTH          1
#define       CLOCK_GATING_HYST            (CLOCK_GATING_EN+CLOCK_GATING_EN_WIDTH)
#define       CLOCK_GATING_HYST_WIDTH        7
// PARTIAL_SEND_WORDS_THR contols the minimum number of 16-byte words of a tile to accumulate in a relay stream before sending it off to the destination.
// If the size of the tile is less than or equal to PARTIAL_SEND_WORDS_THR, then this feild is ignored.
// Default is 16 words
#define       PARTIAL_SEND_WORDS_THR       (CLOCK_GATING_HYST+CLOCK_GATING_HYST_WIDTH)
#define       PARTIAL_SEND_WORDS_THR_WIDTH   8

// AND value of zero masks for the pending message group.
// (Header bits [95:64].)
// Read-only.  Valid only for receiver endpoint streams.
#define   STREAM_MSG_GROUP_ZERO_MASK_AND_REG_INDEX   37

// Returns 1 if the message info register is full (read-only).
#define   STREAM_MSG_INFO_FULL_REG_INDEX   38

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
#define   STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX   39

// Returns 1 if the message info register can accept new message push (read-only).
// Equivalent to checking the condition:
//   (STREAM_MSG_INFO_FULL_REG_INDEX == 0) && (STREAM_MSG_INFO_PTR_REG_INDEX == STREAM_MSG_INFO_WR_PTR_REG_INDEX)
// (I.e. ther is free space in the msg info register, and we dont have any message info headers in the
//  memory buffer about to be fetched.)
#define   STREAM_MSG_INFO_CAN_PUSH_NEW_MSG_REG_INDEX   40

// Concat compress flags from 4 tiles in the pending message group.
// (Header bit 52.)
// Read-only.  Valid only for receiver endpoint streams.
#define   STREAM_MSG_GROUP_COMPRESS_REG_INDEX   41

// Msg_LOCAL_STREAM_CLEAR_NUM specifies the number of messages that should
// be cleared from a gather stream before moving onto the next stream.
// When MSG_ARB_GROUP_SIZE > 1, the order of clearing the streams can be selected
// with MSG_GROUP_STREAM_CLEAR_TYPE. 0 = clear the whole group MSG_LOCAL_STREAM_CLEAR_NUM times,
// 1 = clear each stream of the group MSG_LOCAL_STREAM_CLEAR_NUM times before
// moving onto the next stream in the group.
#define   STREAM_MCAST_GATHER_CLEAR_REG_INDEX   42
#define       MSG_LOCAL_STREAM_CLEAR_NUM        0
#define       MSG_LOCAL_STREAM_CLEAR_NUM_WIDTH    16
#define       MSG_GROUP_STREAM_CLEAR_TYPE       (MSG_LOCAL_STREAM_CLEAR_NUM+MSG_LOCAL_STREAM_CLEAR_NUM_WIDTH)
#define       MSG_GROUP_STREAM_CLEAR_TYPE_WIDTH   1

// Priority for traffic sent to remote destination.
// Valid only for streams capable of remote sending.
// 4-bit value.
// Set to 0 to send traffic under round-robin arbitration.
// Set to 1-15 for priority arbitration (higher values are higher priority).
#define   STREAM_REMOTE_DEST_TRAFFIC_PRIORITY_REG_INDEX   43

// Debug bus stream selection. Write the stream id for the stream that you want exposed on the debug bus
// This register only exists in stream 0.
#define   STREAM_DEBUG_STATUS_SEL_REG_INDEX   44

// Debugging: Non-zero value indicates an invalid stream operation occured.
// Sticky, write 1 to clear.
#define   STREAM_DEBUG_ASSERTIONS_REG_INDEX   45

// Bit mask of connnected local source. Dont care if LOCAL_SOURCES_CONNECTED == 0.
// Mask segments [23:0], [47:24], and [63:48] are at indexes STREAM_LOCAL_SRC_MASK_REG_INDEX,
// STREAM_LOCAL_SRC_MASK_REG_INDEX+1, STREAM_LOCAL_SRC_MASK_REG_INDEX+2.
#define   STREAM_LOCAL_SRC_MASK_REG_INDEX   48

// For receiver endpoint streams that expose the full message header bus to unpacker,
// write this register to specify the full header in case the stream is not snooping
// a remote source but instead also works as a source endpoint.
// Write (STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX+i) to set bits [i*32 +: 32]
// of the message header for the next message, prior to writing STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX.
#define   STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX   60

// Available buffer space at remote destination stream(s).
// Dont care unless REMOTE_RECEIVER == 1.
// Source cant send data unless WORDS_FREE > 0.
// Read-only; updated automatically to maximum value when
// STREAM_REMOTE_DEST_BUF_SIZE_REG is updated.
// For multicast streams, values for successive destinations are at
// subsequent indexes (STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX+1,
// STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX+2, etc.).
#define   STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX   64
#define       REMOTE_DEST_WORDS_FREE       0
#define       REMOTE_DEST_WORDS_FREE_WIDTH   16

// Read-only register view of the bits on the o_full_msg_info bus.
// Exposed as 32-bit read-only registers starting at this index.
#define   STREAM_RECEIVER_MSG_INFO_REG_INDEX   128

// Read-only register that exposes internal states of the stream.
// Useful for debugging. Valid 32-bit data from STREAM_DEBUG_STATUS_REG_INDEX + 0 to STREAM_DEBUG_STATUS_REG_INDEX + 9
#define   STREAM_DEBUG_STATUS_REG_INDEX   224

// Scratch location for firmware usage
// Guarantees that no side-effects occur in Overlay hardware
#define   FIRMWARE_SCRATCH_REG_INDEX   247

// Scratch registers
// Exists only in streams 0-3 and 8-11
// Data can be stored at [23:0] from STREAM_SCRATCH_REG_INDEX + 0 to STREAM_SCRATCH_REG_INDEX + 5
// Can be loaded through overlay blobs.
#define   STREAM_SCRATCH_REG_INDEX   248

#define   STREAM_SCRATCH_0_REG_INDEX   248
#define       NEXT_NRISC_PIC_INT_ON_PHASE       0
#define       NEXT_NRISC_PIC_INT_ON_PHASE_WIDTH   20
#define       NCRISC_TRANS_EN                   (NEXT_NRISC_PIC_INT_ON_PHASE + NEXT_NRISC_PIC_INT_ON_PHASE_WIDTH)
#define       NCRISC_TRANS_EN_WIDTH               1
#define       NCRISC_CMD_ID                     (NCRISC_TRANS_EN + NCRISC_TRANS_EN_WIDTH)
#define       NCRISC_CMD_ID_WIDTH                 3

#define   STREAM_SCRATCH_1_REG_INDEX   249
#define       DRAM_FIFO_RD_PTR_WORDS_LO       0
#define       DRAM_FIFO_RD_PTR_WORDS_LO_WIDTH   24
#define       NCRISC_LOOP_COUNT               0
#define       NCRISC_LOOP_COUNT_WIDTH           24

#define   STREAM_SCRATCH_2_REG_INDEX   250
#define       DRAM_FIFO_RD_PTR_WORDS_HI       0
#define       DRAM_FIFO_RD_PTR_WORDS_HI_WIDTH   4
#define       DRAM_FIFO_WR_PTR_WORDS_LO       (DRAM_FIFO_RD_PTR_WORDS_HI + DRAM_FIFO_RD_PTR_WORDS_HI_WIDTH)
#define       DRAM_FIFO_WR_PTR_WORDS_LO_WIDTH   20
#define       NCRISC_TOTAL_LOOP_ITER          0
#define       NCRISC_TOTAL_LOOP_ITER_WIDTH      24

#define   STREAM_SCRATCH_3_REG_INDEX   251
#define       DRAM_FIFO_WR_PTR_WORDS_HI                 0
#define       DRAM_FIFO_WR_PTR_WORDS_HI_WIDTH             8
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_LO           (DRAM_FIFO_WR_PTR_WORDS_HI + DRAM_FIFO_WR_PTR_WORDS_HI_WIDTH)
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_LO_WIDTH       16
#define       NCRISC_LOOP_INCR                          0
#define       NCRISC_LOOP_INCR_WIDTH                      16
#define       NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES       (NCRISC_LOOP_INCR+NCRISC_LOOP_INCR_WIDTH)
#define       NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES_WIDTH   8

#define   STREAM_SCRATCH_4_REG_INDEX   252
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_HI       0
#define       DRAM_FIFO_CAPACITY_PTR_WORDS_HI_WIDTH   12
#define       DRAM_FIFO_BASE_ADDR_WORDS_LO          (DRAM_FIFO_CAPACITY_PTR_WORDS_HI + DRAM_FIFO_CAPACITY_PTR_WORDS_HI_WIDTH)
#define       DRAM_FIFO_BASE_ADDR_WORDS_LO_WIDTH      12
#define       NCRISC_LOOP_BACK_AUTO_CFG_PTR         0
#define       NCRISC_LOOP_BACK_AUTO_CFG_PTR_WIDTH     24

#define   STREAM_SCRATCH_5_REG_INDEX   253
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

#endif // def NOC_OVERLAY_PARAMETERS_H
