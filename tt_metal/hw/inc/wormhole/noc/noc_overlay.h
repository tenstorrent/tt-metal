// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NOC_OVERLAY_H_
#define _NOC_OVERLAY_H_

#include <stdint.h>
#include <stdbool.h>


/*

  Basic stream semantics:
  =======================


    1. A stream is associated with storage and a flow-control mechanism in
       both directions (i.e. possibility of backpressure on both sender
       and receiver).  A stream ID is unique per processor, and can be
       arbitrarily assigned in the course of initializing an overlay.


    2. A stream data source can be:

         (a) Output of the local processor (i.e. math/packer).

         (b) Data sent by a stream on a remote processor.

         (c) Data sent by another local stream.

       A stream can have at most one source of type (a) or (b), or multiple
       [limited by hardware resources - TBD] sources of type (c).

       For streams that gather data from N sources, it is necessary to set up
       N point-to-point streams, and then set up another stream  with N sources
       of type (c).


     3. A stream destination can be:

         (a) Input to the local processor (i.e., unpacker/math).

         (b) A stream on a single remote processor.

         (c) Streams (each with the same local ID) in a group of
             remote processors that is addressable as a multicast
             destination.  (This may include the local processor
             as well.)  All destinations must have the receiving
             buffer at the same local memory address, and the same
             buffer size for wraparound.

         (d) Another local stream. (As discussed under 2(c) above.)

        A stream can have up to one destination of type (a), up to
        one destination of type (b) or (c), and up to one destination
        of type (d).  Flow control is determined by the behavior of
        the slowest-receiving destination.


     4. After the initialization phase, software needs to deal with only those
        streams that directly consume the output of the local processor (i.e.,
        math/packer) or directly supply the input of the local processor (i.e.,
        math/unpacker).  Everything else runs automatically, providing a simple
        abstraction.


  Example:
  ========

  Suppose we have the following setup for a layer, with 32 cores involved
  (x-dimension size = 4, y-dimension size = 8):

     1. Two clusters, with 16 cores each.  Cluster 0 includes cores with
        coordinates from (0, 0) to (3, 3).  Cluster 1 includes cores with
        coordinates from (0, 4) to (3, 7).

     2. We use z-parallelism only, so ultimately each core's output needs
        to be sent to every other core in both clusters.

     2. Cluster 0 master is at (3, 3).  Cluster 1 master is at (0, 4).

     3. In each cluster, all cores (including master) send their output
        activations to the local cluster's master (the gather operation).

     4. The master multicasts the gathered activations to its own cluster (incl.
        to itself, with data looping back through NOC).  It also needs to send
        the same gathered data to the other master, and to multicast the data
        received from the other master to its local cluster.  (This can't be
        done simply by including the other master into the same multicast that
        delivers data to the local cluster, since the multicast set would be
        non-rectangular.)

  Suppose also that NOC0 is directed so that the shortest path is between nodes
  (x, y)->(x+1, y) or (x, y)->(x, y+1), while NOC1 has the opposite direction.

  We can use the following overlay configuration:

      * Cluster 0 cores use NOC0 unicast to gather data at Master 0.
      * Cluster 1 cores use NOC1 unicast to gather data at Master 1.
      * Master 0 uses NOC1 multicast for output to Cluster 0.
      * Master 1 uses NOC0 multicast for output to Cluster 1.
      * Cluster 0 core (0, 3) uses NOC0 unicast to forward data received
        from Master 0 to Master 1 at (0, 4).
      * Cluster 1 core (3, 4) uses NOC1 unicast to forward data received
        from Master 0 to Master 1 at (3, 3).

  For the N->1 gather from each core to master, the master needs N input streams,
  which are then interleaved into a single output stream.  We need N separate
  stream IDs for these, since each is associated with a separate buffer and a
  separate flow-control mechanism.  Another stream at master node is then used
  to interleave them and output them as multicast.

  In the example, we use stream IDs:
    0      = local math/packer output (at each core)
    1      = local unpacker/math input for the next layer (at each core)
             (this is also forwarded to the other cluster's master from cores (0, 3) and (3, 4))
    2...17 = master gather inputs (from each core)
    18     = master multicast output
    19     = master unicast input forwarded from the other cluster

  The initialization API calls would go as follows (shown for Cluster 0, similar for Cluster 1).

  All non-master cluster 0 cores:

      // Stream 0 = local math/packer output, sent to master
      noc_overlay_stream_set_buf(0, packer_output_buf, packer_output_buf_size, true);
      noc_overlay_stream_set_source_local_producer(0);
      my_index = my_y*4 + my_x;
      my_dest_buf_at_master = master_gather_buf_start + my_index*master_buf_size_per_gather_source;
      noc_overlay_stream_set_dest_remote(0, NOC_XY_ADDR(3, 3, my_dest_buf_at_master0), master_buf_size_per_gather_source, false, 2+my_index);

      // Stream 1 = local math/unpacker input, received from master
      noc_overlay_stream_set_buf(1, unpacker_input_buf, unpacker_input_buf_size, true);
      noc_overlay_stream_set_source_remote(1, 3, 3, 18, my_index);
      noc_overlay_stream_set_dest_local_consumer(1);

  Core (0, 3) only:

      // this adds an extra destination (Master 1) to stream 1, giving it a fanout of 2:
      noc_overlay_stream_set_dest_remote(1, NOC_XY_ADDR(0, 4, dest_buf_at_master1), master1_cluster0_data_buf_size, false, 19);

  Master 0:

      // Stream 18 (master multicast output) is the destination for all input streams, including the local stream 0
      for each (gather_src_x, gather_src_y)
         src_index = gather_src_y*4 + gather_src_x;
         stream_id = 2 + src_index
         noc_overlay_stream_set_buf(stream_id, master_gather_buf_start + src_index*master_buf_size_per_gather_source, gather_input_buf_size, true);
         noc_overlay_stream_set_source_remote(stream_id, gather_src_x, gather_src_y);
         noc_overlay_connect_local_streams(stream_id, 18);

      // Local math/packer output at Master 0 - also connect as one of stream 18 inputs:
      noc_overlay_stream_set_buf(0, packer_output_buf, packer_output_buf_size, true);
      noc_overlay_stream_set_source_local_producer(0);
      noc_overlay_connect_local_streams(0, 18);

      // Remote data from Master 1 via (3, 4)- also connect as one of stream 18 inputs:
      noc_overlay_stream_set_buf(19, master0_cluster1_data_buf, master0_cluster1_data_buf_size, true);
      noc_overlay_stream_set_source_remote(19, 3, 4, 1, 1);
      noc_overlay_connect_local_streams(19, 18);

      // Set multicast destination for stream 18
      noc_overlay_stream_set_dest_remote(18, NOC_MULTICAST_ADDR(0, 0, 3, 3), unpacker_input_buf_size, true, 1);

      // Stream 1 = local math/unpacker input, works the same as on non-master cores (data loops back through NOC router)
      noc_overlay_stream_set_buf(1, unpacker_input_buf, unpacker_input_buf_size, true);
      noc_overlay_stream_set_source_remote(1, 3, 3, 18, 3*4+3);
      noc_overlay_stream_set_dest_local_consumer(1);


  After initialization, everyone calls noc_overlay_stream_start(S) for each initialized stream S.
  This will wait for "init complete" message from all downstream destinations (if any), and then send
  one upstream to all sources (if any).


  During layer execution, the only code that needs to be executed by software is the handling of
  packer data sources and unpacker data destinations (i.e., those streams initialized with the
  set_source_local_producer and set_dest_local_consumer calls).  Everything else runs automatically.

  Local producers/consumers need to use the polling functions below to establish buffer free space/
  data availability and then call data send/receive functions.


  Once the layer is finished, everyone calls noc_overlay_stream_dealloc(S) for each initialized
  stream S.  This will wait for "dealloc" message from all upstream sources (if any), and then
  flush all data from the local buffer and send a "dealloc" message to all downstream destinations
  (if any).  The function is non-blocking, so a polling function is available to test if deallocation
  of each stream's resources is complete.

*/

//  [TBD - API for performance-related settings]
//  [TBD - DRAM destination interleaving mode]


///////////////////////////////
// (1) Initialization functions


/*
  Associate stream ID with a buffer starting address and size.

  Set message_flow_ctrl to true if flow control takes place at message (i.e. tile) level,
  and to false for a raw byte stream.  If true, set message_size_index to the offset of the
  message size field in the message header.

*/
void noc_overlay_stream_set_buf(uint32_t stream_id, uint8_t* buf_start_addr, uint32_t buf_size, bool message_flow_ctrl = true, uint32_t message_size_index = 0);


/*
  Set remote source for the stream ID.

  We need (x, y) coordinates and the (remote) stream ID of the source so the overlay
  logic knows where to send read pointer updates.

  If the source stream is multicast or fans out to a local consumer and remote
  destinations, each remote destination needs to call this function with a unique
  destination index.  (This is because the source must maintain separate read pointers,
  so this index is also necessary to calculate the remote read pointer address).  The
  dest_index argument is don't-care if the source stream has fanout of 1.

 */
void noc_overlay_stream_set_source_remote(uint32_t stream_id, uint32_t source_x, uint32_t source_y, uint32_t source_stream_id, uint32_t dest_index = 0);


/*
  Set local (math/packer) source for the stream ID.

  If this function is called on a stream, it means that data need to be written into
  memory by the local processor, and the local software needs to poll for free buffer
  space and send data using the functions below.

 */
void noc_overlay_stream_set_source_local_producer(uint32_t stream_id);


/*
  Connect two local streams so they act as source and destination.

  A local stream can have an arbitrary [determined by HW resources - TBD] number
  of local streams as input, and only one as the output.

  [TBD - functions to set arbitration policy]

 */
void noc_overlay_connect_local_streams(uint32_t src_stream_id, uint32_t dest_stream_id);


/*
  Set circular arbitration policy for a stream that has multiple connected input
  streams.  (Applicable only to such streams, i.e. to which multiple source streams
  have been connected by calling noc_overlay_connect_local_streams.)

  This function is applicable only to streams working in message mode.

  A stream with circular arbitration and inputs S_1 ... S_N will forward messages only when
  each S_i, i=1...N has a ready message.  Its output data will consist of the messages read
  from S_1...S_N, in that same order.  The downstream receiver can therefore expect to see
  the messages from each source S_i, placed consecutively and in order in its input buffer.

 */
void noc_overlay_stream_circular_arbitration(uint32_t stream_id);


/*
  Set destination for the stream ID. Destination can be either unicast or multicast.

  We need the remote stream ID so the overlay logic knows where to send write pointer updates,
  as well as destination buffer size so it knows how to wrap around the address.

 */
void noc_overlay_stream_set_dest_remote(uint32_t stream_id, uint64_t dest_noc_addr, uint32_t dest_buf_size, bool multicast, uint32_t dest_stream_id = 0);


/*
  Set local (math/unpacker) destination for the stream ID.

  If this function is called on a stream, it means that data need to be read from L1
  by the local processor, and the local software needs to poll for data ready indications
  and receive data when available using the functions below.

 */
void noc_overlay_stream_set_dest_local_consumer(uint32_t stream_id);


/*
  Signal that initialization is done for a stream.  During the overlay initialization
  phase, each processor should initialize all local streams using the above functions,
  and then call the function below with each stream ID in use.

  After this function has been called on all local streams that are used for a layer,
  it is safe to start sending/receiving data.

 */
void noc_overlay_stream_start(uint32_t stream_id);



//////////////////////////
// (2) Data send functions

/*
  There are two mechanisms for data sending:

    (1) For a stream with a dedicated buffer, we can use the following sequence:

        - Call noc_overlay_stream_buf_free_space to find out how much buffer space
          is currently free for sending data.  (Poll if not enough.)

        - Call noc_overlay_stream_buf_curr_write_ptr to get the write pointer where
          data should be written.  Write data to this localtion.

        - Call noc_overlay_stream_send_data to send data down the stream.


    (2) Sender can also provide its own buffer with data to be sent:

        - Call noc_overlay_stream_send_data_buf to initiate sending data from a given
          address.

        - Poll on noc_overlay_stream_data_buf_send_done to ensure data have been sent
          before the sender can discard/overwrite the buffer.

        It is not possible to overlap multiple noc_overlay_stream_send_data_buf operations.
        I.e., each time noc_overlay_stream_send_data_buf is called, the caller must wait
        for the entire buffer to be sent before the memory space can be reused.

*/


/*
  How much free space is there in local data sender stream buffer?
  Needs to be called by software before starting packer.

*/
uint32_t noc_overlay_stream_buf_free_space(uint32_t stream_id);


/*
  Returns the current write pointer for a local outgoing stream.
  Needs to be called by software before starting packer to provide the
  memory address where data should be written.

  (Note: packer needs to handle wraparound, based on the buffer start
  address and size, as specified previously in the noc_overlay_stream_set_buf
  call.)

*/
uint8_t* noc_overlay_stream_buf_curr_write_ptr(uint32_t stream_id);


/*
  Signals that data_size bytes have been written into the buffer and can
  be sent off.

*/
void noc_overlay_stream_send_data(uint32_t stream_id, uint32_t data_size);


/*
  Alternative sending mechanism: send data from a buffer maintained by the
  sender software.

 */
void noc_overlay_stream_send_data_buf(uint32_t stream_id, uint8_t* data_ptr, uint32_t data_size);

/*
  Returns the number of bytes sent with data sent by noc_overlay_stream_send_data_buf
  that have been forwarded so far, so that the corresponding buffer space can be discarded
  or overwritten.

 */
uint32_t noc_overlay_stream_data_buf_send_done(uint32_t stream_id);


/////////////////////////////
// (3) Data receive functions

/*
  Number of received data bytes currently in the buffer.

*/
uint32_t noc_overlay_stream_buf_data_bytes_received(uint32_t stream_id);


/*
  Number of received messages currently in the buffer.

*/
uint32_t noc_overlay_stream_buf_messages_received(uint32_t stream_id);


/*
  Read a word (2 bytes) from the header of the message at the head of the buffer, at the
  given index (number of words) from the start of the message.

*/
uint16_t noc_overlay_stream_buf_message_header_get_word(uint32_t stream_id, uint32_t index);

/*
  Read the size of the message at the head of the buffer.

*/
uint32_t noc_overlay_stream_buf_message_size(uint32_t stream_id);

/*
  Get pointer to the received data/message.

*/
uint8_t* noc_overlay_stream_buf_curr_read_ptr(uint32_t stream_id);


/*
  Signals that data_size bytes have been consumed from the buffer and can
  be discarded.

*/
void noc_overlay_stream_receive_data(uint32_t stream_id, uint32_t data_size);


/*
  Signals that num_msgs messages have been consumed from the buffer and can
  be discarded.

*/
void noc_overlay_stream_receive_messages(uint32_t stream_id, uint32_t num_msgs);



/////////////////////////////////////
// (4) Overlay deallocation functions


/*
  These functions will ensure that all upstream remote sources (if any) have been flushed
  and deallocated, and will subsequently flush all local source streams and propagate the
  deallocation message downstream to all remote destinations (if any).

  The function noc_overlay_stream_dealloc needs to be called for each local stream at the
  end of a layer. It is non-blocking, and software needs to poll noc_overlay_stream_finished
  to ensure that data are flushed and resources deallocated before starting to initialize a
  new overlay (if it reuses any of the same resources).

*/
void noc_overlay_stream_dealloc(uint32_t stream_id);
bool noc_overlay_stream_finished(uint32_t stream_id);


//////

/*

  Structures for formatting NOC stream auto config.

*/


struct noc_stream_cfg_header_struct {
  unsigned phase_num_incr : 12;
  unsigned curr_phase_num_msgs: 12;
  unsigned num_cfg_reg_writes: 8;
};

struct noc_stream_cfg_reg_write_struct {
  unsigned reg_index : 8;
  unsigned reg_val: 24;
};

typedef struct noc_stream_cfg_header_struct noc_stream_cfg_header;

typedef struct noc_stream_cfg_reg_write_struct noc_stream_cfg_reg_write;


#endif //ndef _NOC_OVERLAY_H_
