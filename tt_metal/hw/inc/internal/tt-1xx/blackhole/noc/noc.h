// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NOC_H_
#define _NOC_H_

#include <stdint.h>
#include <stdbool.h>

//////

#include "noc_parameters.h"

/*

  Basic NOC API

  Common arguments:

   * bool linked: all sequences of function calls with “linked” set to true
     targeting the same destination will manifest on the NoC as a single
     multi-command packet, guaranteeing they complete in-order.  For commands
     targeting different destinations, it is not possible to provide this
     guarantee.

     Moreover, it is not possible to provide linked ordering between different
     unicast/multicast classes of NOC virtual channels.

   * unicast addresses: all unicast address arguments are given as 40-bit
     addresses (type uint64_t):
        - bits [31:0] = byte offset in local L1 memory,
        - bits [35:32]/[39:36] = X/Y coordinate prefixes.

     The addresses can be provided using the above macro NOC_XY_ADDR.  For
     example, address 0x1000 in L1 of Tensix (1, 2) can be given as
     NOC_XY_ADDR(1, 2, 0x1000).

   * multicast addresses: all multicast address arguments are given as a 48-bit
     combination of 32-bit local address and coordinates of the upper left and
     lower right corners of the multicast rectangle.

     The address can be provided using the macro NOC_MULTICAST_ADDR above.
     For example, using NOC_MULTICAST_ADDR(1, 4, 6, 5) will multicast to
     12 destinations, i.e. all those with X coordinates between 1 and 6 and
     Y-coordinates between 4 and 5 (inclusive).

   All addresses are in the form of byte offsets, regardless of the minimal
   access granularity.  Address bits below the minimal access granularity are
   ignored.

*/

/*
  Copy data from source to destination address.  Supports narrow transfers
  (size not a multiple of 16 bytes).  However, the alignment of source and
  destination start addresses (i.e., bits [3:0]) must be identical.  If the
  alignment is not identical, the one from destination address is assumed.

  If copying from local memory to a remote destination, set posted=false to
  request an ack that will increment the NIU_MST_WR_ACK_RECEIVED counter.
  This value can be compared with NIU_MST_NONPOSTED_WR_REQ_STARTED to ensure
  the writes are flushed.  (Note that copying more than NOC_MAX_BURST_SIZE
  triggers multiple underlying NOC requests.)

  If src_addr is remote, the request is always posted and the parameter is
  ignored.

  <src_coordinate> => NOC ID portion of source address (unicast)
  <src_addr> => source address (unicast)
  <dst_coordinate> => NOC ID portion of destination address (unicast)
  <dst_addr> => destination address (unicast)
  <size> => number of bytes to copy
  <linked> => link with previous call for ordering
  <posted> => if copying from a local address, avoid sending ack on the
              response channel
  <static_vc_alloc> => use static VC allocation
  <static_vc> => use VC 0/1 for static request; don't-care if static_vc_alloc=0
  <vc_arb_priority> =>  arbitration priority for VC allocation;
              set to 0 disable arbitration priority & use round-robin always
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_copy(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint32_t vc_arb_priority,
    uint8_t transaction_id);

/*
  Copy data from source to destination address and accumulate at destination.
  Supports narrow transfers (size not a multiple of 16 bytes).
  However, the alignment of source and destination start addresses (i.e., bits [3:0])
  must be identical.  If the alignment is not identical, the one from destination
  address is assumed.

  If copying from local memory to a remote destination, set posted=false to
  request an ack that will increment the NIU_MST_WR_ACK_RECEIVED counter.
  This value can be compared with NIU_MST_NONPOSTED_WR_REQ_STARTED to ensure
  the writes are flushed.  (Note that copying more than NOC_MAX_BURST_SIZE
  triggers multiple underlying NOC requests.)

  If src_addr is remote, the request is always posted and the parameter is
  ignored.

  <src_coordinate> => NOC ID portion of source address (unicast)
  <src_addr> => source address (unicast)
  <dst_coordinate> => NOC ID portion of destination address (unicast)
  <dst_addr> => destination address (unicast)
  <size> => number of bytes to copy
  <linked> => link with previous call for ordering
  <posted> => if copying from a local address, avoid sending ack on the
              response channel
  <static_vc_alloc> => use static VC allocation
  <static_vc> => use VC 0/1 for static request; don't-care if static_vc_alloc=0
  <vc_arb_priority> =>  arbitration priority for VC allocation;
              set to 0 disable arbitration priority & use round-robin always
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
  <data_format => The format of the data used for accumulation: NOC_AT_ACC_*, e.g. NOC_AT_ACC_FP32>
  <disable_saturation => Set to disable accumulation saturation, such that values will wrap around if the addition
  result cannot fit.>
*/
void noc_accumulate(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    bool multicast,
    uint32_t multicast_mode,
    uint32_t vc_arb_priority,
    uint8_t transaction_id,
    uint8_t data_format,
    bool disable_saturation);

/*
  Copy a single word with byte-enables from source to destination address.
  Works similar to noc_copy, except limited to a single-word transfer and
  provides the option to specify partial byte-enables.

  This call works only with transfers from local memory.

  <src_coordinate> => NOC ID portion of source address (unicast)
  <src_addr> => source address (unicast, must be local memory)
  <dst_coordinate> => NOC ID portion of destination address (unicast)
  <dst_addr> => destination address (unicast)
  <be> => byte enable mask
  <linked> => link with previous call for ordering
  <posted> => if copying from a local address, avoid sending ack on the
              response channel
  <static_vc_alloc> => use static VC allocation
  <static_vc> => use VC 0/1 for static request; don't-care if static_vc_alloc=0
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_copy_word_be(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint64_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id);

/*
  Write a single 32-bit value using inline header data. (The effect is the
  same as when writing with noc_copy, however such writes save the bandwidth
  of an additional flit for register access.)

  <dst_coordinate> => NOC ID portion of destination address (unicast)
  <dst_addr> => destination address (unicast)
  <be> => byte enable mask
  <linked> => link with previous call for ordering
  <posted> => if copying from a local address, avoid sending ack on the
              response channel
  <static_vc_alloc> => use static VC allocation
  <static_vc> => use VC 0/1 for static request; don't-care if static_vc_alloc=0
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_write_dw_inline(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t val,
    uint8_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id);

/*
  Copy data from source to multiple destinations via multicast.  Supports
  narrow transfers (size not a multiple of 16 bytes).  However, the alignment
  of source and destination start addresses (i.e., bits [3:0]) must be identical.
  If the alignment is not identical, the one from destination address is assumed.

  If copying from local memory and posted=false, a separate ack is received from
  each destination.

  <src_coordinate> => NOC ID portion of source address (unicast)
  <src_addr> => source address (unicast)
  <dst_coordinate> => NOC ID portion of destination address (multicast)
  <dst_addr> => destination address (multicast)
  <multicast_mode> => multicast direction (0 = x-direction, 1 = y-direction)
  <size> => number of bytes to copy
  <linked> => link with previous call for ordering
  <posted> => if copying from a local address, avoid sending ack on the
              response channel
  <static_vc_alloc> => use static VC allocation
  <static_vc> => use VC 0/1 for static request; don't-care if static_vc_alloc=0
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_multicast_copy(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id);

// support multicast ability to exclude nodes
void noc_multicast_copy_exclude(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint32_t multicast_exclude,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id);

// support src include
void noc_multicast_copy_src_include(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id);

/*
  Multicast version of noc_copy_word_be.

  Like noc_copy_word_be, this call works only with transfers from local memory,
  and is limited to single-word transfers.

  <src_coordinate> => NOC ID portion of source address (unicast)
  <src_addr> => source address (unicast)
  <dst_coordinate> => NOC ID portion of destination address (multicast)
  <dst_addr> => destination address (multicast)
  <multicast_mode> => multicast direction (0 = x-direction, 1 = y-direction)
  <be> => byte enable mask
  <linked> => link with previous call for ordering
  <posted> => if copying from a local address, avoid sending ack on the
              response channel
  <static_vc_alloc> => use static VC allocation
  <static_vc> => use VC 0/1 for static request; don't-care if static_vc_alloc=0
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_multicast_copy_word_be(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint64_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id);

/*
  Multicast version of noc_write_dw_inline.

  <dst_coordinate> => NOC ID portion of destination address (unicast)
  <dst_addr> => destination address (unicast)
  <multicast_mode> => multicast direction (0 = x-direction, 1 = y-direction)
  <be> => byte enable mask
  <linked> => link with previous call for ordering
  <posted> => if copying from a local address, avoid sending ack on the
              response channel
  <static_vc_alloc> => use static VC allocation
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_multicast_write_dw_inline(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t val,
    uint32_t multicast_mode,
    uint8_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id);

/*
  Atomic wrapping increment of 32-bit value at destination address.  The address has
  4-byte granularity.  The increment result wraps around the address aligned relative
  to the specified wrapping size.   Increment is an arbitrary value, while wrapping
  limit is calculated from the given argument as 2^(<wrap>+1).  (Therefore, for 32-bit
  values, setting <wrap>=31 implies no wrapping except the 32-bit integer maximum.)

  For example, if:
      wrap = 7 (wrap to 0x100),
      incr = 0x80 (increase by 0x80),
      current value = 0x21C0,

  then the aligned valud is 0x2100, and the new value is:
      0x2100 + ((0x1C0 + 0x80) % 0x100) = 0x2140.

  <noc_coordinate> => NOC ID portion of addr (unicast)
  <addr> => counter address (unicast)
  <incr> => increment
  <wrap> => log2(wrapping limit)-1
  <linked> => link with previous call for ordering
*/
void noc_atomic_increment(uint32_t noc_coordinate, uint64_t addr, uint32_t incr, uint32_t wrap, bool linked);

/*
  Performs the same operation as noc_atomic_increment and reads the previous value from the
  destination address to <read_addr>.  The <read_addr> address also has 4-byte granularity,
  and the return value updates only the corresponding 32 bits in local memory.

  There is no alignment requirement between <addr> and <read_addr>.

  This function can be used to reserve space in a remote buffer by operating on the write
  pointer.

  The status of the returned read can be determined by calling noc_atomic_read_updates_completed
  (see below).

  <noc_coordinate> => NOC ID portion of addr (unicast)
  <addr> => counter address (unicast)
  <incr> => increment
  <wrap> => log2(wrapping limit)-1
  <read_coordinate> => NOC ID portion of address
  <read_addr> => address to store the previous value
  <linked> => link with previous call for ordering
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_atomic_read_and_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap,
    uint32_t read_coordinate,
    uint64_t read_addr,
    bool linked,
    uint8_t transaction_id);

/*
  Performs the same operation as noc_atomic_increment on multiple multicast destinations.

  <noc_coordinate> => NOC ID portion of addr (multicast)
  <addr> => counter address (multicast)
  <multicast_mode> => multicast direction (0 = x-direction, 1 = y-direction)
  <incr> => increment
  <wrap> => log2(wrapping limit)-1
  <linked> => link with previous call for ordering
*/
void noc_multicast_atomic_increment(
    uint32_t noc_coordinate, uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, bool linked);

/*
  Performs the same operation as noc_atomic_read_and_increment on multiple multicast destinations.

  Each destination returns the previous value, and the final value written to read_addr is undefined,
  depending on the order in which updates are delivered.  Therefore, the 32-bit value at this address
  must be reserved to be modified by this call, but its final value should be ignored.

  The value returned by noc_atomic_read_updates_completed will increment with each returned response.
  The intended use case for this function is to perform atomic increments at multiple destinations, and
  subsequently call noc_atomic_read_updates_completed to ensure all the updates have completed.

  <noc_coordinate> => NOC ID portion of addr (multicast)
  <addr> => counter address (multicast)
  <multicast_mode> => multicast direction (0 = x-direction, 1 = y-direction)
  <incr> => increment
  <wrap> => log2(wrapping limit)-1
  <read_coordinate> => NOC ID portion of read_addr (multicast)
  <read_addr> => address to store the previous value
  <linked> => link with previous call for ordering
  <transaction_id => optional ID tag for the outgoing request (0-15, used for
              selective transaction flush)>
*/
void noc_multicast_atomic_read_and_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    uint32_t multicast_mode,
    uint32_t incr,
    uint32_t wrap,
    uint32_t read_coordinate,
    uint64_t read_addr,
    bool linked,
    uint8_t transaction_id);

/*
  Set command buffer ID (0-3) to use for the next commmand issued.
*/
void noc_set_cmd_buf(uint32_t cmd_buf_id);

/*
  Get current setting for command buffer ID (0-3).
*/
uint32_t noc_get_cmd_buf();

/*
  Set NOC instance (0-1) to use for the next commmand issued.
*/
void noc_set_active_instance(uint32_t noc_id);

/*
  Get current setting for NOC instance (0-1) to use.
*/
uint32_t noc_get_active_instance();

/*
  Returns the number of atomic operations that return a value (such as
  atomic_read_and_increment) that have completed since the last reset event.

  The counter is 32-bit and wraps when the number exceeds 2^32-1.

  By tracking this value and polling for its increase, firmware can determine
  that the remote pointer updates have completed and the responses have been
  committed to local memory.
 */
uint32_t noc_atomic_read_updates_completed();

/*
  Returns the number of write acks received

  The counter is 32-bit and wraps when the number exceeds 2^32-1.

  By tracking this value and polling for its increase, firmware can determine
  that the noc copy operations have completed.
 */

volatile uint32_t noc_wr_ack_received();

/*
  Returns the number of read responses received

  The counter is 32-bit and wraps when the number exceeds 2^32-1.

  By tracking this value and polling for its increase, firmware can determine
  that the noc copy operations (from remote to local) have completed.
 */

volatile uint32_t noc_rd_resp_received();
/*
  Returns true if the active command buffer is presently available (i.e., no pending
  request that is being backpressured by the NOC).

  Issuing a command while the command buffer is busy results in undefined behavior.

  All above functions use command buffer 0 and spin on noc_command_ready() at entry,
  so there is no need to call it explicitly before these calls.

*/
bool noc_command_ready();

/*
  Returns ID & dateline info of the local node in the format:
     {10'b0, i_dateline_node_y[0:0], i_dateline_node_x[0:0],
     i_local_nocid[3:0],
     i_noc_y_size[3:0], i_noc_x_size[3:0],
     i_local_nodeid_y[3:0], i_local_nodeid_x[3:0]}
*/
uint32_t noc_local_node_id();

/*
  Returns value of specific status register (see noc_parameters.h for the list).
*/
uint32_t noc_status_reg(uint32_t status_reg_id);

/*
  Sets value of specific NOC config register (see noc_parameters.h for the list).
*/
void noc_set_cfg_reg(uint32_t cfg_reg_id, uint32_t val);

/*
  Gets value of specific NOC config register (see noc_parameters.h for the list).
*/
uint32_t noc_get_cfg_reg(uint32_t cfg_reg_id);

/*
  Reset to 0 each transaction ID outstanding request counter for which the corresponding
  id_mask bit is set.
*/
void noc_clear_req_id_cnt(uint32_t id_mask);

//////////////////////////////////////////////////////////////////
//////////////////////// ECC Functions ///////////////////////////
//////////////////////////////////////////////////////////////////

/*
  Allows for the enabling/disabling of ECC features in the NIU and Routers
  Enabling full ECC is a two stage process. First you must call noc_ecc_cfg_stage_1 for all tensix, sync (ensuring all
  writes went through), and then call noc_ecc_cfg_stage_2 for all tensix.
*/
void noc_ecc_cfg_stage_1(bool header_ckh_bits_en);

/*
  Allows for the enabling/disabling of ECC features in the NIU and Routers
  Enabling full ECC is a two stage process. First you must call noc_ecc_cfg_stage_1 for all tensix, sync (ensuring all
  writes went through), and then call noc_ecc_cfg_stage_2 for all tensix.
*/
void noc_ecc_cfg_stage_2(
    bool niu_mem_parity_en,
    bool router_mem_parity_en,
    bool header_secded_en,
    bool mem_parity_int_en,
    bool header_sec_int_en,
    bool header_ded_int_en);

/*
  Clears the corresponding ECC error interrupt and number of errors register
*/
void noc_ecc_clear_err(bool clear_mem_parity_err, bool clear_header_sec, bool clear_header_ded);

/*
  Increments the corresponding number of errors register by 1.
  Debug use only.
*/
void noc_ecc_force_err(bool force_mem_parity_err, bool force_header_sec, bool force_header_ded);

/*
  Gets the number of memory parity errors. This is the sum of the number of parity errors in the router and niu memories
  (if enabled in noc_ecc_cfg()). This register indicates a fatal error in the system.
*/
uint32_t noc_ecc_get_num_mem_parity_errs();

/*
  Gets the number of single errors that were corrected in the header. This register should be treated as a warning of
  system instability.
*/
uint32_t noc_ecc_get_num_header_sec();

/*
  Gets the number of double errors detected in the header. This register indicates a fatal error in the system.
*/
uint32_t noc_ecc_get_num_header_ded();

//////

#endif  // ndef _NOC_H_
