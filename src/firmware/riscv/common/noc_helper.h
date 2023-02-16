#pragma once

#define OVERLAY_STREAM_BUF_FREE_SPACE(core,stream_id) \
  ( core->read_stream_register(stream_id, STREAM_BUF_SPACE_AVAILABLE_REG_INDEX) * MEM_WORD_BYTES )

#define OVERLAY_STREAM_BUF_MESSAGES_RECEIVED(core,stream_id) \
  ( core->read_stream_register(stream_id, STREAM_NUM_MSGS_RECEIVED_REG_INDEX) )

#define OVERLAY_STREAM_BUF_BASE(core,stream_id) \
  ( core->template l1_cast<uint8_t*>(core->read_stream_register(stream_id, STREAM_BUF_START_REG_INDEX) * MEM_WORD_BYTES) )

#define OVERLAY_STREAM_BUF_SIZE(core,stream_id) \
  ( core->read_stream_register(stream_id, STREAM_BUF_SIZE_REG_INDEX) * MEM_WORD_BYTES )

#define OVERLAY_STREAM_BUF_LIMIT(core,stream_id) \
  ( OVERLAY_STREAM_BUF_BASE(core,stream_id) + OVERLAY_STREAM_BUF_SIZE(core,stream_id) )

#define OVERLAY_STREAM_BUF_CURR_WRITE_PTR(core,stream_id) \
  ( OVERLAY_STREAM_BUF_BASE(core,stream_id) + core->read_stream_register(stream_id, STREAM_WR_PTR_REG_INDEX) * MEM_WORD_BYTES )

#define OVERLAY_STREAM_TILE_HEADER_BUF_CURR_WRITE_ADDR_16B(core,stream_id) \
  ( core->read_stream_register(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX) )

#define OVERLAY_STREAM_RECEIVER_MSG_INFO_REG_INDEX(core,stream_id,word_index) \
    ( core->read_stream_register(stream_id, STREAM_RECEIVER_MSG_INFO_REG_INDEX + (word_index)) )

#define OVERLAY_STREAM_MSG_PTR(core,stream_id) \
    ( core->template l1_cast<uint8_t*>(core->read_stream_register(stream_id, STREAM_RECEIVER_MSG_INFO_REG_INDEX + 0) * MEM_WORD_BYTES ) )

#define OVERLAY_STREAM_BUF_CURR_READ_PTR(core,stream_id) \
  ( OVERLAY_STREAM_BUF_BASE(core,stream_id) + core->read_stream_register(stream_id, STREAM_RD_PTR_REG_INDEX) * MEM_WORD_BYTES )

#define OVERLAY_STREAM_PUSH_TO_MSG_INFO(core,stream_id,msg_addr,msg_size) \
  ( core->write_stream_register(stream_id, STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX, \
                                ((((msg_addr) / MEM_WORD_BYTES) & ((1 << SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH)-1)) << SOURCE_ENDPOINT_NEW_MSG_ADDR) | \
                                ((((msg_size) / MEM_WORD_BYTES) & ((1 << SOURCE_ENDPOINT_NEW_MSG_SIZE_WIDTH)-1)) << SOURCE_ENDPOINT_NEW_MSG_SIZE)) )

#define OVERLAY_STREAM_SET_MSG_HEADER(core,stream_id,header) \
    core->write_stream_register(stream_id, STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX + 0, \
      (header->tile_id << 17) | (header->tile_size_16B)); \
    core->write_stream_register(stream_id, STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX + 1, \
      (header->format << 16) | (header->reserved_1 << 8) | (header->metadata_size_16B)); \
    core->write_stream_register(stream_id, STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX + 2, \
      header->zero_mask); \
    core->write_stream_register(stream_id, STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX + 3, \
      header->reserved_3);

 // cannot specify address higher than 1MB via STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX
#define OVERLAY_STREAM_SEND_DATA(core, stream_id, msg_addr, msg_size, header) \
  if (core->read_stream_register(stream_id,                                   \
                                 STREAM_MSG_INFO_FULL_REG_INDEX) == 0 &&      \
      core->read_stream_register(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX) == \
          core->read_stream_register(stream_id,                               \
                                     STREAM_MSG_INFO_WR_PTR_REG_INDEX) &&     \
      msg_addr <                                                              \
          ((1 << SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH) * MEM_WORD_BYTES)) {     \
    OVERLAY_STREAM_SET_MSG_HEADER(core, stream_id, header);                   \
    OVERLAY_STREAM_PUSH_TO_MSG_INFO(core, stream_id, msg_addr, msg_size);     \
  } else {                                                                    \
    core->write_stream_register(                                              \
        stream_id, STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX,                    \
        ((1 & ((1 << SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH) - 1))                \
         << SOURCE_ENDPOINT_NEW_MSGS_NUM) |                                   \
            ((((msg_size) / MEM_WORD_BYTES) &                                 \
              ((1 << SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE_WIDTH) - 1))         \
             << SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE));                        \
  }

#define OVERLAY_STREAM_RECEIVE_MESSAGES(core,stream_id,num_msgs) \
  core->write_stream_register(stream_id, STREAM_MSG_INFO_CLEAR_REG_INDEX, num_msgs); \
  core->write_stream_register(stream_id, STREAM_MSG_DATA_CLEAR_REG_INDEX, num_msgs)

#define OVERLAY_STREAM_MSG_CLEAR_SET(core,stream_id) \
  ( core->read_stream_register(stream_id, STREAM_MSG_DATA_CLEAR_REG_INDEX) )

#define OVERLAY_STREAM_NUM_MSGS_RECEIVED_INC(core,stream_id) \
  ( core->read_stream_register(stream_id, STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX) )
