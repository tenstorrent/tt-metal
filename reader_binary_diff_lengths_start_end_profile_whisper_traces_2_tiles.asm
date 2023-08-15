
/home/ubuntu/tt-metal/built_kernels/eltwise_binary_writer_unary_reader_binary_diff_lengths/5086620033119610385/brisc/brisc.elf:     file format elf32-littleriscv


Disassembly of section .init:

00000a00 <_start>:

CRT_START:
  # Initialize global pointer
  .option push
  .option norelax
1:auipc gp, %pcrel_hi(__global_pointer$)
 a00:	ffb00197          	auipc	gp, 0xffb00
  addi  gp, gp, %pcrel_lo(1b)
 a04:	e0018193          	addi	gp, gp, -512 # ffb00800 <__global_pointer$+0x0>

/* set stack pointer */
  la sp, __stack_top
 a08:	ffb00117          	auipc	sp, 0xffb00
 a0c:	ca810113          	addi	sp, sp, -856 # ffb006b0 <__global_pointer$+0xfffffeb0>

  # Clear the bss segment
  la      a0, __l1_bss_start
 a10:	ffaff517          	auipc	a0, 0xffaff
 a14:	5f050513          	addi	a0, a0, 1520 # ffb00000 <__global_pointer$+0xfffff800>
  la      a1, __l1_bss_end
 a18:	ffaff597          	auipc	a1, 0xffaff
 a1c:	60c58593          	addi	a1, a1, 1548 # ffb00024 <__global_pointer$+0xfffff824>
  call    wzerorange
 a20:	00001097          	auipc	ra, 0x1
 a24:	b50080e7          	jalr	-1200(ra) # 1570 <wzerorange>

  la      a0, __ldm_bss_start
 a28:	ffaff517          	auipc	a0, 0xffaff
 a2c:	5fc50513          	addi	a0, a0, 1532 # ffb00024 <__global_pointer$+0xfffff824>
  la      a1, __ldm_bss_end
 a30:	ffb00597          	auipc	a1, 0xffb00
 a34:	88058593          	addi	a1, a1, -1920 # ffb002b0 <__global_pointer$+0xfffffab0>
  call    wzerorange
 a38:	00001097          	auipc	ra, 0x1
 a3c:	b38080e7          	jalr	-1224(ra) # 1570 <wzerorange>

  la      s2, __init_array_start
 a40:	ffaff917          	auipc	s2, 0xffaff
 a44:	5c090913          	addi	s2, s2, 1472 # ffb00000 <__global_pointer$+0xfffff800>
  la      s3, __init_array_end
 a48:	ffaff997          	auipc	s3, 0xffaff
 a4c:	5b898993          	addi	s3, s3, 1464 # ffb00000 <__global_pointer$+0xfffff800>
  j       2f
 a50:	0100006f          	j	a60 <_start+0x60>
1:lw      a0, 0(s2)
 a54:	00092503          	lw	a0, 0(s2)
  jalr    a0
 a58:	000500e7          	jalr	a0
  addi    s2, s2, 4
 a5c:	00490913          	addi	s2, s2, 4
2:bne     s2, s3, 1b
 a60:	ff391ae3          	bne	s2, s3, a54 <_start+0x54>
   * sp+0: argv[0] -> sp+8
   * sp+4: argv[1] = NULL
   * sp+8: s1
   * sp+c: 0
   */
  addi    sp, sp, -16 /* (stack is aligned to 16 bytes in riscv calling convention) */
 a64:	ff010113          	addi	sp, sp, -16
  addi    a0, sp, 8
 a68:	00810513          	addi	a0, sp, 8
  sw      a0, 0(sp)
 a6c:	00a12023          	sw	a0, 0(sp)
  sw      zero, 4(sp)
 a70:	00012223          	sw	zero, 4(sp)
  sw      s1, 8(sp)
 a74:	00912423          	sw	s1, 8(sp)
  sw      zero, 12(sp)
 a78:	00012623          	sw	zero, 12(sp)

  li      a0, 1 # argc = 1
 a7c:	00100513          	li	a0, 1
  mv      a1, sp
 a80:	00010593          	mv	a1, sp
  mv      a2, zero
 a84:	00000613          	li	a2, 0

  call    main
 a88:	018000ef          	jal	ra, aa0 <main>
  tail    exit
 a8c:	2d50006f          	j	1560 <exit>

00000a90 <_fini>:
  .global _fini
  .type   _fini, @function
_init:
_fini:
  # These don't have to do anything since we use init_array/fini_array.
  ret
 a90:	00008067          	ret

Disassembly of section .text:

00000aa0 <main>:
     aa0:	fc010113          	addi	sp, sp, -64
     * */
    inline __attribute__((always_inline)) void init_BR_profiler()
    {
#if defined(PROFILE_KERNEL) && defined(COMPILE_FOR_BRISC)
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_NC);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
     aa4:	0001b7b7          	lui	a5, 0x1b
     aa8:	02812c23          	sw	s0, 56(sp)
     aac:	02912a23          	sw	s1, 52(sp)
     ab0:	02112e23          	sw	ra, 60(sp)
     ab4:	03212823          	sw	s2, 48(sp)
     ab8:	03312623          	sw	s3, 44(sp)
     abc:	03412423          	sw	s4, 40(sp)
     ac0:	03512223          	sw	s5, 36(sp)
     ac4:	03612023          	sw	s6, 32(sp)
     ac8:	01712e23          	sw	s7, 28(sp)
     acc:	01812c23          	sw	s8, 24(sp)
     ad0:	01912a23          	sw	s9, 20(sp)
     ad4:	00200693          	li	a3, 2
     ad8:	80078713          	addi	a4, a5, -2048 # 1a800 <substitutes.cpp.8dc51291+0x12fa3>
     adc:	00d72023          	sw	a3, 0(a4)
        buffer [DROPPED_MARKER_COUNTER] = 0;
     ae0:	00072223          	sw	zero, 4(a4)
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_T0);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
     ae4:	88078613          	addi	a2, a5, -1920
     ae8:	04d62623          	sw	a3, 76(a2)
     aec:	8cc78713          	addi	a4, a5, -1844
        buffer [DROPPED_MARKER_COUNTER] = 0;
     af0:	00072223          	sw	zero, 4(a4)
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_T1);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
     af4:	98078613          	addi	a2, a5, -1664
     af8:	00d62c23          	sw	a3, 24(a2)
     afc:	99878713          	addi	a4, a5, -1640
        buffer [DROPPED_MARKER_COUNTER] = 0;
     b00:	00072223          	sw	zero, 4(a4)
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_T2);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
     b04:	a0078613          	addi	a2, a5, -1536
     b08:	06d62223          	sw	a3, 100(a2)
     b0c:	a6478713          	addi	a4, a5, -1436
        buffer [DROPPED_MARKER_COUNTER] = 0;
     b10:	00072223          	sw	zero, 4(a4)
        buffer = reinterpret_cast<uint32_t*>(get_debug_print_buffer());
     b14:	b3078613          	addi	a2, a5, -1232
        wIndex = MARKER_DATA_START;
     b18:	80d1ae23          	sw	a3, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
        buffer = reinterpret_cast<uint32_t*>(get_debug_print_buffer());
     b1c:	82c1a023          	sw	a2, -2016(gp) # ffb00020 <_ZN15kernel_profiler6bufferE>
        buffer [BUFFER_END_INDEX] = wIndex;
     b20:	b0078793          	addi	a5, a5, -1280
     b24:	02d7a823          	sw	a3, 48(a5)
        buffer [DROPPED_MARKER_COUNTER] = 0;
     b28:	00062223          	sw	zero, 4(a2)
}

inline uint32_t reg_read_barrier(uint32_t addr)
{
    volatile uint32_t *p_reg = reinterpret_cast<volatile uint32_t *> (addr);
    uint32_t data = p_reg[0];
     b2c:	ffb12737          	lui	a4, 0xffb12
     b30:	1f072503          	lw	a0, 496(a4) # ffb121f0 <__global_pointer$+0x119f0>
    local_mem_barrier = data;
     b34:	80a1ac23          	sw	a0, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
     b38:	1f872583          	lw	a1, 504(a4)
#endif

        // Either buffer has room for more markers or the end of FW marker is place on the last marker spot
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
            ((timer_id == CC_MAIN_END) && !((wIndex + TIMER_DATA_UINT32_SIZE) > (PRINT_BUFFER_SIZE/sizeof(uint32_t))))) {
	    buffer[wIndex+TIMER_ID] = timer_id;
     b3c:	00100713          	li	a4, 1
    local_mem_barrier = data;
     b40:	80b1ac23          	sw	a1, -2024(gp) # ffb00018 <local_mem_barrier>
     b44:	02e7ac23          	sw	a4, 56(a5)
	    buffer[wIndex+TIMER_VAL_L] = time_L;
     b48:	81c1a683          	lw	a3, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
     b4c:	00168713          	addi	a4, a3, 1
     b50:	00271713          	slli	a4, a4, 0x2
     b54:	00c70833          	add	a6, a4, a2
	    buffer[wIndex+TIMER_VAL_H] = time_H;
     b58:	00470713          	addi	a4, a4, 4
     b5c:	00c70733          	add	a4, a4, a2
	    buffer[wIndex+TIMER_VAL_L] = time_L;
     b60:	00a82023          	sw	a0, 0(a6)
	    buffer[wIndex+TIMER_VAL_H] = time_H;
     b64:	00b72023          	sw	a1, 0(a4)
            wIndex += TIMER_DATA_UINT32_SIZE;
     b68:	00368713          	addi	a4, a3, 3
     b6c:	80e1ae23          	sw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
            buffer [BUFFER_END_INDEX] = wIndex;
     b70:	02e7a823          	sw	a4, 48(a5)
     b74:	ffb306b7          	lui	a3, 0xffb30
     b78:	100007b7          	lui	a5, 0x10000
     b7c:	10f6a623          	sw	a5, 268(a3) # ffb3010c <__global_pointer$+0x2f90c>


inline uint32_t NOC_CMD_BUF_READ_REG(uint32_t noc, uint32_t buf, uint32_t addr) {
  uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
  volatile uint32_t* ptr = (volatile uint32_t*)offset;
  return *ptr;
     b80:	ffb207b7          	lui	a5, 0xffb20
     b84:	02c7a783          	lw	a5, 44(a5) # ffb2002c <__global_pointer$+0x1f82c>
     b88:	00c7d713          	srli	a4, a5, 0xc
     b8c:	07f77713          	andi	a4, a4, 127
     b90:	0137d793          	srli	a5, a5, 0x13
     b94:	80e18a23          	sb	a4, -2028(gp) # ffb00014 <noc_size_x>
     b98:	07f7f793          	andi	a5, a5, 127
     b9c:	80f18aa3          	sb	a5, -2027(gp) # ffb00015 <noc_size_y>
     ba0:	02c6a783          	lw	a5, 44(a3)
     ba4:	00100793          	li	a5, 1
     ba8:	09402703          	lw	a4, 148(zero) # 94 <__firmware_stack_size-0x36c>
     bac:	fef71ee3          	bne	a4, a5, ba8 <main+0x108>
     bb0:	ffb487b7          	lui	a5, 0xffb48
// only BRISC to call this
void init_sync_registers() {

    volatile uint* tiles_received_ptr;
    volatile uint* tiles_acked_ptr;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
     bb4:	ffb68737          	lui	a4, 0xffb68
     bb8:	01078793          	addi	a5, a5, 16 # ffb48010 <__global_pointer$+0x47810>
     bbc:	000016b7          	lui	a3, 0x1
     bc0:	01070713          	addi	a4, a4, 16 # ffb68010 <__global_pointer$+0x67810>
      tiles_received_ptr = get_cb_tiles_received_ptr(operand);
      tiles_received_ptr[0] = 0;
     bc4:	0007a023          	sw	zero, 0(a5)
      tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
      tiles_acked_ptr[0] = 0;
     bc8:	fe07ae23          	sw	zero, -4(a5)
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
     bcc:	00d787b3          	add	a5, a5, a3
     bd0:	fee79ae3          	bne	a5, a4, bc4 <main+0x124>
}

// can be used on NCRICS and/or BRISC, as both can act as tile producers into Tensix
void setup_cb_read_write_interfaces() {

  volatile std::uint32_t* circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);
     bd4:	0001a737          	lui	a4, 0x1a

  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
     bd8:	0001a7b7          	lui	a5, 0x1a
  volatile std::uint32_t* circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);
     bdc:	40070713          	addi	a4, a4, 1024 # 1a400 <substitutes.cpp.8dc51291+0x12ba3>
  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
     be0:	58078693          	addi	a3, a5, 1408 # 1a580 <substitutes.cpp.8dc51291+0x12d23>

    // write_to_local_mem_barrier are needed on GS because of the RTL bug
    // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
    std::uint32_t fifo_addr = circular_buffer_config_addr[0];
     be4:	00072603          	lw	a2, 0(a4)
    std::uint32_t fifo_size = circular_buffer_config_addr[1];
     be8:	00472603          	lw	a2, 4(a4)
    std::uint32_t fifo_size_tiles = circular_buffer_config_addr[2];
     bec:	00872603          	lw	a2, 8(a4)
    cb_write_interface[cb_id].fifo_limit = fifo_addr + fifo_size - 1;  // to check if we need to wrap
    cb_write_interface[cb_id].fifo_wr_ptr = fifo_addr;
    cb_write_interface[cb_id].fifo_size = fifo_size;
    cb_write_interface[cb_id].fifo_size_tiles = fifo_size_tiles;

    circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG; // move by 3 uint32's
     bf0:	00c70713          	addi	a4, a4, 12
    local_mem_barrier = data;
     bf4:	80c1ac23          	sw	a2, -2024(gp) # ffb00018 <local_mem_barrier>
  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
     bf8:	fed716e3          	bne	a4, a3, be4 <main+0x144>
  }

  circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);

  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
     bfc:	0001a6b7          	lui	a3, 0x1a
     c00:	82418713          	addi	a4, gp, -2012 # ffb00024 <cb_read_interface>
  circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);
     c04:	40078793          	addi	a5, a5, 1024
     c08:	82418913          	addi	s2, gp, -2012 # ffb00024 <cb_read_interface>
  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
     c0c:	58068693          	addi	a3, a3, 1408 # 1a580 <substitutes.cpp.8dc51291+0x12d23>

    // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
    std::uint32_t fifo_addr = circular_buffer_config_addr[0];
     c10:	0007a503          	lw	a0, 0(a5)
    std::uint32_t fifo_size = circular_buffer_config_addr[1];
     c14:	0047a603          	lw	a2, 4(a5)

    cb_read_interface[cb_id].fifo_limit = fifo_addr + fifo_size - 1;  // to check if we need to wrap
    cb_read_interface[cb_id].fifo_rd_ptr = fifo_addr;
    cb_read_interface[cb_id].fifo_size = fifo_size;

    circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG; // move by 3 uint32's
     c18:	00c78793          	addi	a5, a5, 12
    cb_read_interface[cb_id].fifo_rd_ptr = fifo_addr;
     c1c:	00a72423          	sw	a0, 8(a4)
    cb_read_interface[cb_id].fifo_limit = fifo_addr + fifo_size - 1;  // to check if we need to wrap
     c20:	00c505b3          	add	a1, a0, a2
     c24:	fff58593          	addi	a1, a1, -1
     c28:	00b72223          	sw	a1, 4(a4)
    cb_read_interface[cb_id].fifo_size = fifo_size;
     c2c:	00c72023          	sw	a2, 0(a4)
    local_mem_barrier = data;
     c30:	80c1ac23          	sw	a2, -2024(gp) # ffb00018 <local_mem_barrier>
  for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
     c34:	01070713          	addi	a4, a4, 16
     c38:	fcd79ce3          	bne	a5, a3, c10 <main+0x170>
* This file can be autogenerated by running generate_l1_bank_id_reshuffled_map.py.
* This file contains values that are visible to both host and device compiled code.
*/

void init_shuffled_l1_bank_id_mapping(uint8_t shuffled_bank_ids[]) {
	shuffled_bank_ids[0] = 48;
     c3c:	486c3737          	lui	a4, 0x486c3
     c40:	20090793          	addi	a5, s2, 512
     c44:	b3070713          	addi	a4, a4, -1232 # 486c2b30 <substitutes.cpp.8dc51291+0x486bb2d3>
     c48:	00e7a023          	sw	a4, 0(a5)
	shuffled_bank_ids[1] = 43;
	shuffled_bank_ids[2] = 108;
	shuffled_bank_ids[3] = 72;
	shuffled_bank_ids[4] = 29;
     c4c:	17361737          	lui	a4, 0x17361
     c50:	e1d70713          	addi	a4, a4, -483 # 17360e1d <substitutes.cpp.8dc51291+0x173595c0>
     c54:	00e7a223          	sw	a4, 4(a5)
	shuffled_bank_ids[5] = 14;
	shuffled_bank_ids[6] = 54;
	shuffled_bank_ids[7] = 23;
	shuffled_bank_ids[8] = 124;
     c58:	2c0a6737          	lui	a4, 0x2c0a6
     c5c:	e7c70713          	addi	a4, a4, -388 # 2c0a5e7c <substitutes.cpp.8dc51291+0x2c09e61f>
     c60:	00e7a423          	sw	a4, 8(a5)
	shuffled_bank_ids[9] = 94;
	shuffled_bank_ids[10] = 10;
	shuffled_bank_ids[11] = 44;
	shuffled_bank_ids[12] = 3;
     c64:	6f533737          	lui	a4, 0x6f533
     c68:	10370713          	addi	a4, a4, 259 # 6f533103 <substitutes.cpp.8dc51291+0x6f52b8a6>
     c6c:	00e7a623          	sw	a4, 12(a5)
	shuffled_bank_ids[13] = 49;
	shuffled_bank_ids[14] = 83;
	shuffled_bank_ids[15] = 111;
	shuffled_bank_ids[16] = 16;
     c70:	656b6737          	lui	a4, 0x656b6
     c74:	f1070713          	addi	a4, a4, -240 # 656b5f10 <substitutes.cpp.8dc51291+0x656ae6b3>
     c78:	00e7a823          	sw	a4, 16(a5)
	shuffled_bank_ids[17] = 95;
	shuffled_bank_ids[18] = 107;
	shuffled_bank_ids[19] = 101;
	shuffled_bank_ids[20] = 110;
     c7c:	7e041737          	lui	a4, 0x7e041
     c80:	66e70713          	addi	a4, a4, 1646 # 7e04166e <substitutes.cpp.8dc51291+0x7e039e11>
     c84:	00e7aa23          	sw	a4, 20(a5)
	shuffled_bank_ids[21] = 22;
	shuffled_bank_ids[22] = 4;
	shuffled_bank_ids[23] = 126;
	shuffled_bank_ids[24] = 25;
     c88:	434c0737          	lui	a4, 0x434c0
     c8c:	21970713          	addi	a4, a4, 537 # 434c0219 <substitutes.cpp.8dc51291+0x434b89bc>
     c90:	00e7ac23          	sw	a4, 24(a5)
	shuffled_bank_ids[25] = 2;
	shuffled_bank_ids[26] = 76;
	shuffled_bank_ids[27] = 67;
	shuffled_bank_ids[28] = 46;
     c94:	554b3737          	lui	a4, 0x554b3
     c98:	42e70713          	addi	a4, a4, 1070 # 554b342e <substitutes.cpp.8dc51291+0x554abbd1>
     c9c:	00e7ae23          	sw	a4, 28(a5)
	shuffled_bank_ids[29] = 52;
	shuffled_bank_ids[30] = 75;
	shuffled_bank_ids[31] = 85;
	shuffled_bank_ids[32] = 89;
     ca0:	0f567737          	lui	a4, 0xf567
     ca4:	05970713          	addi	a4, a4, 89 # f567059 <substitutes.cpp.8dc51291+0xf55f7fc>
     ca8:	02e7a023          	sw	a4, 32(a5)
	shuffled_bank_ids[33] = 112;
	shuffled_bank_ids[34] = 86;
	shuffled_bank_ids[35] = 15;
	shuffled_bank_ids[36] = 57;
     cac:	5b3a7737          	lui	a4, 0x5b3a7
     cb0:	83970713          	addi	a4, a4, -1991 # 5b3a6839 <substitutes.cpp.8dc51291+0x5b39efdc>
     cb4:	02e7a223          	sw	a4, 36(a5)
	shuffled_bank_ids[37] = 104;
	shuffled_bank_ids[38] = 58;
	shuffled_bank_ids[39] = 91;
	shuffled_bank_ids[40] = 50;
     cb8:	732f2737          	lui	a4, 0x732f2
     cbc:	53270713          	addi	a4, a4, 1330 # 732f2532 <substitutes.cpp.8dc51291+0x732eacd5>
     cc0:	02e7a423          	sw	a4, 40(a5)
	shuffled_bank_ids[41] = 37;
	shuffled_bank_ids[42] = 47;
	shuffled_bank_ids[43] = 115;
	shuffled_bank_ids[44] = 13;
     cc4:	78155737          	lui	a4, 0x78155
     cc8:	40d70713          	addi	a4, a4, 1037 # 7815540d <substitutes.cpp.8dc51291+0x7814dbb0>
     ccc:	02e7a623          	sw	a4, 44(a5)
	shuffled_bank_ids[45] = 84;
	shuffled_bank_ids[46] = 21;
	shuffled_bank_ids[47] = 120;
	shuffled_bank_ids[48] = 99;
     cd0:	23766737          	lui	a4, 0x23766
     cd4:	86370713          	addi	a4, a4, -1949 # 23765863 <substitutes.cpp.8dc51291+0x2375e006>
     cd8:	02e7a823          	sw	a4, 48(a5)
	shuffled_bank_ids[49] = 88;
	shuffled_bank_ids[50] = 118;
	shuffled_bank_ids[51] = 35;
	shuffled_bank_ids[52] = 19;
     cdc:	67450737          	lui	a4, 0x67450
     ce0:	61370713          	addi	a4, a4, 1555 # 67450613 <substitutes.cpp.8dc51291+0x67448db6>
     ce4:	02e7aa23          	sw	a4, 52(a5)
	shuffled_bank_ids[53] = 6;
	shuffled_bank_ids[54] = 69;
	shuffled_bank_ids[55] = 103;
	shuffled_bank_ids[56] = 80;
     ce8:	7b3b1737          	lui	a4, 0x7b3b1
     cec:	45070713          	addi	a4, a4, 1104 # 7b3b1450 <substitutes.cpp.8dc51291+0x7b3a9bf3>
     cf0:	02e7ac23          	sw	a4, 56(a5)
	shuffled_bank_ids[57] = 20;
	shuffled_bank_ids[58] = 59;
	shuffled_bank_ids[59] = 123;
	shuffled_bank_ids[60] = 121;
     cf4:	7f492737          	lui	a4, 0x7f492
     cf8:	27970713          	addi	a4, a4, 633 # 7f492279 <substitutes.cpp.8dc51291+0x7f48aa1c>
     cfc:	02e7ae23          	sw	a4, 60(a5)
	shuffled_bank_ids[61] = 34;
	shuffled_bank_ids[62] = 73;
	shuffled_bank_ids[63] = 127;
	shuffled_bank_ids[64] = 30;
     d00:	08182737          	lui	a4, 0x8182
     d04:	c1e70713          	addi	a4, a4, -994 # 8181c1e <substitutes.cpp.8dc51291+0x817a3c1>
     d08:	04e7a023          	sw	a4, 64(a5)
	shuffled_bank_ids[65] = 28;
	shuffled_bank_ids[66] = 24;
	shuffled_bank_ids[67] = 8;
	shuffled_bank_ids[68] = 41;
     d0c:	3f5c2737          	lui	a4, 0x3f5c2
     d10:	f2970713          	addi	a4, a4, -215 # 3f5c1f29 <substitutes.cpp.8dc51291+0x3f5ba6cc>
     d14:	04e7a223          	sw	a4, 68(a5)
	shuffled_bank_ids[69] = 31;
	shuffled_bank_ids[70] = 92;
	shuffled_bank_ids[71] = 63;
	shuffled_bank_ids[72] = 0;
     d18:	010b7737          	lui	a4, 0x10b7
     d1c:	70070713          	addi	a4, a4, 1792 # 10b7700 <substitutes.cpp.8dc51291+0x10afea3>
     d20:	04e7a423          	sw	a4, 72(a5)
	shuffled_bank_ids[73] = 119;
	shuffled_bank_ids[74] = 11;
	shuffled_bank_ids[75] = 1;
	shuffled_bank_ids[76] = 82;
     d24:	427a0737          	lui	a4, 0x427a0
     d28:	75270713          	addi	a4, a4, 1874 # 427a0752 <substitutes.cpp.8dc51291+0x42798ef5>
     d2c:	04e7a623          	sw	a4, 76(a5)
	shuffled_bank_ids[77] = 7;
	shuffled_bank_ids[78] = 122;
	shuffled_bank_ids[79] = 66;
	shuffled_bank_ids[80] = 56;
     d30:	1a467737          	lui	a4, 0x1a467
     d34:	23870713          	addi	a4, a4, 568 # 1a467238 <substitutes.cpp.8dc51291+0x1a45f9db>
     d38:	04e7a823          	sw	a4, 80(a5)
	shuffled_bank_ids[81] = 114;
	shuffled_bank_ids[82] = 70;
	shuffled_bank_ids[83] = 26;
	shuffled_bank_ids[84] = 81;
     d3c:	37285737          	lui	a4, 0x37285
     d40:	e5170713          	addi	a4, a4, -431 # 37284e51 <substitutes.cpp.8dc51291+0x3727d5f4>
     d44:	04e7aa23          	sw	a4, 84(a5)
	shuffled_bank_ids[85] = 78;
	shuffled_bank_ids[86] = 40;
	shuffled_bank_ids[87] = 55;
	shuffled_bank_ids[88] = 125;
     d48:	3c477737          	lui	a4, 0x3c477
     d4c:	47d70713          	addi	a4, a4, 1149 # 3c47747d <substitutes.cpp.8dc51291+0x3c46fc20>
     d50:	04e7ac23          	sw	a4, 88(a5)
	shuffled_bank_ids[89] = 116;
	shuffled_bank_ids[90] = 71;
	shuffled_bank_ids[91] = 60;
	shuffled_bank_ids[92] = 42;
     d54:	5d095737          	lui	a4, 0x5d095
     d58:	72a70713          	addi	a4, a4, 1834 # 5d09572a <substitutes.cpp.8dc51291+0x5d08decd>
     d5c:	04e7ae23          	sw	a4, 92(a5)
	shuffled_bank_ids[93] = 87;
	shuffled_bank_ids[94] = 9;
	shuffled_bank_ids[95] = 93;
	shuffled_bank_ids[96] = 105;
     d60:	4d122737          	lui	a4, 0x4d122
     d64:	76970713          	addi	a4, a4, 1897 # 4d122769 <substitutes.cpp.8dc51291+0x4d11af0c>
     d68:	06e7a023          	sw	a4, 96(a5)
	shuffled_bank_ids[97] = 39;
	shuffled_bank_ids[98] = 18;
	shuffled_bank_ids[99] = 77;
	shuffled_bank_ids[100] = 90;
     d6c:	66204737          	lui	a4, 0x66204
     d70:	45a70713          	addi	a4, a4, 1114 # 6620445a <substitutes.cpp.8dc51291+0x661fcbfd>
     d74:	06e7a223          	sw	a4, 100(a5)
	shuffled_bank_ids[101] = 68;
	shuffled_bank_ids[102] = 32;
	shuffled_bank_ids[103] = 102;
	shuffled_bank_ids[104] = 79;
     d78:	6d601737          	lui	a4, 0x6d601
     d7c:	c4f70713          	addi	a4, a4, -945 # 6d600c4f <substitutes.cpp.8dc51291+0x6d5f93f2>
     d80:	06e7a423          	sw	a4, 104(a5)
	shuffled_bank_ids[105] = 12;
	shuffled_bank_ids[106] = 96;
	shuffled_bank_ids[107] = 109;
	shuffled_bank_ids[108] = 36;
     d84:	1b401737          	lui	a4, 0x1b401
     d88:	12470713          	addi	a4, a4, 292 # 1b401124 <substitutes.cpp.8dc51291+0x1b3f98c7>
     d8c:	06e7a623          	sw	a4, 108(a5)
	shuffled_bank_ids[109] = 17;
	shuffled_bank_ids[110] = 64;
	shuffled_bank_ids[111] = 27;
	shuffled_bank_ids[112] = 74;
     d90:	263d3737          	lui	a4, 0x263d3
     d94:	d4a70713          	addi	a4, a4, -694 # 263d2d4a <substitutes.cpp.8dc51291+0x263cb4ed>
     d98:	06e7a823          	sw	a4, 112(a5)
	shuffled_bank_ids[113] = 45;
	shuffled_bank_ids[114] = 61;
	shuffled_bank_ids[115] = 38;
	shuffled_bank_ids[116] = 106;
     d9c:	33756737          	lui	a4, 0x33756
     da0:	46a70713          	addi	a4, a4, 1130 # 3375646a <substitutes.cpp.8dc51291+0x3374ec0d>
     da4:	06e7aa23          	sw	a4, 116(a5)
	shuffled_bank_ids[117] = 100;
	shuffled_bank_ids[118] = 117;
	shuffled_bank_ids[119] = 51;
	shuffled_bank_ids[120] = 62;
     da8:	05214737          	lui	a4, 0x5214
     dac:	13e70713          	addi	a4, a4, 318 # 521413e <substitutes.cpp.8dc51291+0x520c8e1>
     db0:	06e7ac23          	sw	a4, 120(a5)
	shuffled_bank_ids[121] = 65;
	shuffled_bank_ids[122] = 33;
	shuffled_bank_ids[123] = 5;
	shuffled_bank_ids[124] = 53;
     db4:	62617737          	lui	a4, 0x62617
     db8:	13570713          	addi	a4, a4, 309 # 62617135 <substitutes.cpp.8dc51291+0x6260f8d8>
     dbc:	06e7ae23          	sw	a4, 124(a5)
     dc0:	28090793          	addi	a5, s2, 640
     dc4:	ffe40737          	lui	a4, 0xffe40
     dc8:	00e7a023          	sw	a4, 0(a5)
     dcc:	ffe50737          	lui	a4, 0xffe50
     dd0:	00e7a223          	sw	a4, 4(a5)
     dd4:	ffe60737          	lui	a4, 0xffe60
     dd8:	00e7a423          	sw	a4, 8(a5)
  return ptr[0];
     ddc:	ffb12737          	lui	a4, 0xffb12
     de0:	1b072783          	lw	a5, 432(a4) # ffb121b0 <__global_pointer$+0x119b0>
     de4:	ffb00a37          	lui	s4, 0xffb00
     de8:	0607da63          	bgez	a5, e5c <main+0x3bc>
     dec:	000a2783          	lw	a5, 0(s4) # ffb00000 <__global_pointer$+0xfffff800>
     df0:	ffb206b7          	lui	a3, 0xffb20
     df4:	02c68693          	addi	a3, a3, 44 # ffb2002c <__global_pointer$+0x1f82c>
     df8:	01079793          	slli	a5, a5, 0x10
     dfc:	00d787b3          	add	a5, a5, a3
     e00:	0007a683          	lw	a3, 0(a5)

inline uint memory_read(uint addr)
{
#ifndef MODELT
    volatile uint * buf = reinterpret_cast<volatile uint * >(addr);
    return buf[0];
     e04:	1f072603          	lw	a2, 496(a4)
     e08:	1f872583          	lw	a1, 504(a4)
     e0c:	0066d793          	srli	a5, a3, 0x6
     e10:	03f7f713          	andi	a4, a5, 63
     e14:	00171793          	slli	a5, a4, 0x1
     e18:	00e787b3          	add	a5, a5, a4
     e1c:	03f6f713          	andi	a4, a3, 63
     e20:	00279793          	slli	a5, a5, 0x2
     e24:	ff370713          	addi	a4, a4, -13
     e28:	00e78733          	add	a4, a5, a4
     e2c:	00171793          	slli	a5, a4, 0x1
     e30:	00e787b3          	add	a5, a5, a4
     e34:	00b79793          	slli	a5, a5, 0xb
     e38:	00c78733          	add	a4, a5, a2
     e3c:	00f737b3          	sltu	a5, a4, a5
     e40:	00b787b3          	add	a5, a5, a1
     e44:	ffb126b7          	lui	a3, 0xffb12
     e48:	1f06a583          	lw	a1, 496(a3) # ffb121f0 <__global_pointer$+0x119f0>
     e4c:	1f86a603          	lw	a2, 504(a3)
     e50:	fef66ce3          	bltu	a2, a5, e48 <main+0x3a8>
     e54:	00c79463          	bne	a5, a2, e5c <main+0x3bc>
     e58:	fee5e8e3          	bltu	a1, a4, e48 <main+0x3a8>
  ptr[0] = val;
     e5c:	03f00793          	li	a5, 63
     e60:	ffb119b7          	lui	s3, 0xffb11
     e64:	02f9a223          	sw	a5, 36(s3) # ffb11024 <__global_pointer$+0x10824>
     e68:	00000513          	li	a0, 0
     e6c:	000a2023          	sw	zero, 0(s4)
     e70:	6c8000ef          	jal	ra, 1538 <_Z15noc_get_cfg_regm>
     e74:	00156593          	ori	a1, a0, 1
     e78:	00000513          	li	a0, 0
     e7c:	694000ef          	jal	ra, 1510 <_Z15noc_set_cfg_regmm>
     e80:	00100513          	li	a0, 1
     e84:	6b4000ef          	jal	ra, 1538 <_Z15noc_get_cfg_regm>
     e88:	00156593          	ori	a1, a0, 1
     e8c:	00100513          	li	a0, 1
     e90:	680000ef          	jal	ra, 1510 <_Z15noc_set_cfg_regmm>
     e94:	00100a93          	li	s5, 1
     e98:	00000513          	li	a0, 0
     e9c:	015a2023          	sw	s5, 0(s4)
     ea0:	698000ef          	jal	ra, 1538 <_Z15noc_get_cfg_regm>
     ea4:	00156593          	ori	a1, a0, 1
     ea8:	00000513          	li	a0, 0
     eac:	664000ef          	jal	ra, 1510 <_Z15noc_set_cfg_regmm>
     eb0:	00100513          	li	a0, 1
     eb4:	684000ef          	jal	ra, 1538 <_Z15noc_get_cfg_regm>
     eb8:	00156593          	ori	a1, a0, 1
     ebc:	00100513          	li	a0, 1
     ec0:	650000ef          	jal	ra, 1510 <_Z15noc_set_cfg_regmm>
     ec4:	ffef07b7          	lui	a5, 0xffef0
     ec8:	28078793          	addi	a5, a5, 640 # ffef0280 <__global_pointer$+0x3efa80>
     ecc:	000a2023          	sw	zero, 0(s4)
     ed0:	ffc00737          	lui	a4, 0xffc00
     ed4:	04e7ac23          	sw	a4, 88(a5)
     ed8:	0000e737          	lui	a4, 0xe
     edc:	a0070713          	addi	a4, a4, -1536 # da00 <substitutes.cpp.8dc51291+0x61a3>
     ee0:	04e7a423          	sw	a4, 72(a5)
     ee4:	00013737          	lui	a4, 0x13
     ee8:	a0070713          	addi	a4, a4, -1536 # 12a00 <substitutes.cpp.8dc51291+0xb1a3>
     eec:	04e7a623          	sw	a4, 76(a5)
     ef0:	00017737          	lui	a4, 0x17
     ef4:	a0070713          	addi	a4, a4, -1536 # 16a00 <substitutes.cpp.8dc51291+0xf1a3>
     ef8:	04e7a823          	sw	a4, 80(a5)
     efc:	00700713          	li	a4, 7
     f00:	04e7aa23          	sw	a4, 84(a5)
}

extern "C" void wzerorange(uint32_t *start, uint32_t *end);
inline void wzeromem(uint32_t start, uint32_t len)
{
    wzerorange((uint32_t *)start, (uint32_t *)(start + len));
     f04:	00001537          	lui	a0, 0x1
     f08:	0557ae23          	sw	s5, 92(a5)
     f0c:	a0050593          	addi	a1, a0, -1536 # a00 <_start>
     f10:	80050513          	addi	a0, a0, -2048
     f14:	65c000ef          	jal	ra, 1570 <wzerorange>
     f18:	000197b7          	lui	a5, 0x19
     f1c:	0007a783          	lw	a5, 0(a5) # 19000 <substitutes.cpp.8dc51291+0x117a3>
     f20:	04078c63          	beqz	a5, f78 <main+0x4d8>
    buf[0] = value;
     f24:	5a000793          	li	a5, 1440
     f28:	00f9a023          	sw	a5, 0(s3)
     f2c:	000047b7          	lui	a5, 0x4
     f30:	00f9a223          	sw	a5, 4(s3)
     f34:	40000793          	li	a5, 1024
     f38:	00f9a423          	sw	a5, 8(s3)
     f3c:	0159a623          	sw	s5, 12(s3)
     f40:	04000793          	li	a5, 64
     f44:	00f9a823          	sw	a5, 16(s3)
    return buf[0];
     f48:	0149a783          	lw	a5, 20(s3)
     f4c:	ffb116b7          	lui	a3, 0xffb11
     f50:	00800713          	li	a4, 8
     f54:	00f12623          	sw	a5, 12(sp)
     f58:	0146a783          	lw	a5, 20(a3) # ffb11014 <__global_pointer$+0x10814>
     f5c:	00f12623          	sw	a5, 12(sp)
     f60:	00c12783          	lw	a5, 12(sp)
     f64:	0097f793          	andi	a5, a5, 9
     f68:	fee798e3          	bne	a5, a4, f58 <main+0x4b8>
     f6c:	ffb127b7          	lui	a5, 0xffb12
     f70:	00007737          	lui	a4, 0x7
     f74:	1ae7a823          	sw	a4, 432(a5) # ffb121b0 <__global_pointer$+0x119b0>
     f78:	ffef07b7          	lui	a5, 0xffef0
     f7c:	28078713          	addi	a4, a5, 640 # ffef0280 <__global_pointer$+0x3efa80>
     f80:	00f00693          	li	a3, 15
     f84:	04d72223          	sw	a3, 68(a4) # 7044 <noc.c.715ded3a+0x2b4>
  instrn_buffer[0] = instrn;
     f88:	ffe406b7          	lui	a3, 0xffe40
     f8c:	10180737          	lui	a4, 0x10180
     f90:	00e6a023          	sw	a4, 0(a3) # ffe40000 <__global_pointer$+0x33f800>
     f94:	8a003737          	lui	a4, 0x8a003
     f98:	00a70713          	addi	a4, a4, 10 # 8a00300a <__global_pointer$+0x8a50280a>
     f9c:	00e6a023          	sw	a4, 0(a3)
     fa0:	02000737          	lui	a4, 0x2000
     fa4:	00e6a023          	sw	a4, 0(a3)
  uint32_t cfg_data = cfg_regs[addr];
     fa8:	0087a703          	lw	a4, 8(a5)
  cfg_data |= wrdata;
     fac:	40000637          	lui	a2, 0x40000
     fb0:	00c76733          	or	a4, a4, a2
  cfg_regs[addr] = rmw_cfg_value(cfg_shamt, cfg_mask, wr_val, cfg_data);
     fb4:	00e7a423          	sw	a4, 8(a5)
  uint32_t cfg_data = cfg_regs[addr];
     fb8:	0087a703          	lw	a4, 8(a5)
  cfg_data |= wrdata;
     fbc:	80000637          	lui	a2, 0x80000
     fc0:	00c76733          	or	a4, a4, a2
  cfg_regs[addr] = rmw_cfg_value(cfg_shamt, cfg_mask, wr_val, cfg_data);
     fc4:	00e7a423          	sw	a4, 8(a5)
  uint32_t cfg_data = cfg_regs[addr];
     fc8:	00c7a583          	lw	a1, 12(a5)
  cfg_data |= wrdata;
     fcc:	00001637          	lui	a2, 0x1
     fd0:	ffe60613          	addi	a2, a2, -2 # ffe <main+0x55e>
     fd4:	2005c713          	xori	a4, a1, 512
     fd8:	00c77733          	and	a4, a4, a2
     fdc:	00b74733          	xor	a4, a4, a1
  cfg_regs[addr] = rmw_cfg_value(cfg_shamt, cfg_mask, wr_val, cfg_data);
     fe0:	00e7a623          	sw	a4, 12(a5)
  instrn_buffer[0] = instrn;
     fe4:	a31007b7          	lui	a5, 0xa3100
     fe8:	00878793          	addi	a5, a5, 8 # a3100008 <__global_pointer$+0xa35ff808>
     fec:	00f6a023          	sw	a5, 0(a3)
     ff0:	19c00713          	li	a4, 412
     ff4:	09c00793          	li	a5, 156
     ff8:	0007a023          	sw	zero, 0(a5)
     ffc:	00478793          	addi	a5, a5, 4
    1000:	fee79ce3          	bne	a5, a4, ff8 <main+0x558>
    return buf[0];
    1004:	ffb12737          	lui	a4, 0xffb12
    1008:	1f072583          	lw	a1, 496(a4) # ffb121f0 <__global_pointer$+0x119f0>
    100c:	1f872783          	lw	a5, 504(a4)
    1010:	ffb206b7          	lui	a3, 0xffb20
    1014:	08b02c23          	sw	a1, 152(zero) # 98 <__firmware_stack_size-0x368>
    1018:	08f02e23          	sw	a5, 156(zero) # 9c <__firmware_stack_size-0x364>
    101c:	02c6a603          	lw	a2, 44(a3) # ffb2002c <__global_pointer$+0x1f82c>
  *ptr = val;
    1020:	40068593          	addi	a1, a3, 1024
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_MID, (uint32_t)(xy_local_addr >> 32));

    noc_reads_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    noc_nonposted_writes_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    noc_nonposted_writes_acked[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
    1024:	ffb00537          	lui	a0, 0xffb00
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    1028:	00665793          	srli	a5, a2, 0x6
    102c:	03f7f793          	andi	a5, a5, 63
    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    1030:	03f67613          	andi	a2, a2, 63
    uint64_t xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);
    1034:	00679793          	slli	a5, a5, 0x6
    1038:	00c7e7b3          	or	a5, a5, a2
  *ptr = val;
    103c:	00f6a223          	sw	a5, 4(a3)
    1040:	ffb21637          	lui	a2, 0xffb21
    1044:	80f62223          	sw	a5, -2044(a2) # ffb20804 <__global_pointer$+0x20004>
    1048:	00002637          	lui	a2, 0x2
    104c:	09060613          	addi	a2, a2, 144 # 2090 <brisc.cc.d630dca8+0x41e>
    1050:	00c5ae23          	sw	a2, 28(a1)
    1054:	00f5a823          	sw	a5, 16(a1)
  return *ptr;
    1058:	20068693          	addi	a3, a3, 512
    105c:	0086a783          	lw	a5, 8(a3)
    1060:	0286a783          	lw	a5, 40(a3)
    noc_nonposted_writes_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    1064:	ffb005b7          	lui	a1, 0xffb00
    1068:	00f5a223          	sw	a5, 4(a1) # ffb00004 <__global_pointer$+0xfffff804>
  return *ptr;
    106c:	0046a783          	lw	a5, 4(a3)
    1070:	00458593          	addi	a1, a1, 4
    noc_nonposted_writes_acked[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
    1074:	00f52623          	sw	a5, 12(a0) # ffb0000c <__global_pointer$+0xfffff80c>
    1078:	000197b7          	lui	a5, 0x19
    107c:	0047a783          	lw	a5, 4(a5) # 19004 <substitutes.cpp.8dc51291+0x117a7>
    1080:	00c50513          	addi	a0, a0, 12
    1084:	02078663          	beqz	a5, 10b0 <main+0x610>
  return ptr[0];
    1088:	18070793          	addi	a5, a4, 384
    108c:	0307a703          	lw	a4, 48(a5)
}

inline void assert_trisc_reset() {
  uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
  uint32_t trisc_reset_mask = 0x7000;
  WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 | trisc_reset_mask);
    1090:	000076b7          	lui	a3, 0x7
    1094:	00d76733          	or	a4, a4, a3
  ptr[0] = val;
    1098:	02e7a823          	sw	a4, 48(a5)
  return ptr[0];
    109c:	0307a703          	lw	a4, 48(a5)


inline void deassert_trisc_reset() {
  uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
  uint32_t trisc_reset_mask = 0x7000;
  WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 & ~trisc_reset_mask);
    10a0:	ffff96b7          	lui	a3, 0xffff9
    10a4:	fff68693          	addi	a3, a3, -1 # ffff8fff <__global_pointer$+0x4f87ff>
    10a8:	00d77733          	and	a4, a4, a3
  ptr[0] = val;
    10ac:	02e7a823          	sw	a4, 48(a5)
    10b0:	000017b7          	lui	a5, 0x1
    10b4:	59078793          	addi	a5, a5, 1424 # 1590 <__local_mem_rodata_end_addr>
    10b8:	fff00737          	lui	a4, 0xfff00
    10bc:	00e7f733          	and	a4, a5, a4
    10c0:	ffb006b7          	lui	a3, 0xffb00
    10c4:	04d71a63          	bne	a4, a3, 1118 <main+0x678>
    10c8:	ffb00737          	lui	a4, 0xffb00
    10cc:	00070713          	mv	a4, a4
    10d0:	40e787b3          	sub	a5, a5, a4
    10d4:	0027d793          	srli	a5, a5, 0x2
    10d8:	04078063          	beqz	a5, 1118 <main+0x678>
    10dc:	00007737          	lui	a4, 0x7
    10e0:	e8070713          	addi	a4, a4, -384 # 6e80 <noc.c.715ded3a+0xf0>
    10e4:	00e787b3          	add	a5, a5, a4
    10e8:	ffae46b7          	lui	a3, 0xffae4
    10ec:	0001c737          	lui	a4, 0x1c
    10f0:	00279793          	slli	a5, a5, 0x2
    10f4:	a0070713          	addi	a4, a4, -1536 # 1ba00 <substitutes.cpp.8dc51291+0x141a3>
    10f8:	60068693          	addi	a3, a3, 1536 # ffae4600 <__global_pointer$+0xfffe3e00>
    10fc:	00072803          	lw	a6, 0(a4)
    1100:	00d70633          	add	a2, a4, a3
    1104:	00470713          	addi	a4, a4, 4
    1108:	01062023          	sw	a6, 0(a2)
    110c:	fee798e3          	bne	a5, a4, 10fc <main+0x65c>
    1110:	ffc7a783          	lw	a5, -4(a5)
    1114:	80f1ac23          	sw	a5, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
    1118:	ffb127b7          	lui	a5, 0xffb12
    111c:	1f07a303          	lw	t1, 496(a5) # ffb121f0 <__global_pointer$+0x119f0>
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
    1120:	81c1a703          	lw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
    1124:	0001b837          	lui	a6, 0x1b
    local_mem_barrier = data;
    1128:	8061ac23          	sw	t1, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
    112c:	1f87a883          	lw	a7, 504(a5)
    1130:	00670613          	addi	a2, a4, 6
    1134:	03200693          	li	a3, 50
    local_mem_barrier = data;
    1138:	8111ac23          	sw	a7, -2024(gp) # ffb00018 <local_mem_barrier>
    113c:	b3080793          	addi	a5, a6, -1232 # 1ab30 <substitutes.cpp.8dc51291+0x132d3>
    1140:	1ec6e463          	bltu	a3, a2, 1328 <main+0x888>
	    buffer[wIndex+TIMER_ID] = timer_id;
    1144:	00271693          	slli	a3, a4, 0x2
    1148:	00d78633          	add	a2, a5, a3
    114c:	00200e13          	li	t3, 2
    1150:	01c62023          	sw	t3, 0(a2)
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    1154:	00468613          	addi	a2, a3, 4
    1158:	00c78633          	add	a2, a5, a2
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    115c:	00868693          	addi	a3, a3, 8
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    1160:	00662023          	sw	t1, 0(a2)
            wIndex += TIMER_DATA_UINT32_SIZE;
    1164:	00370713          	addi	a4, a4, 3
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    1168:	00d787b3          	add	a5, a5, a3
    116c:	0117a023          	sw	a7, 0(a5)
            wIndex += TIMER_DATA_UINT32_SIZE;
    1170:	80e1ae23          	sw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
            buffer [BUFFER_END_INDEX] = wIndex;
    1174:	b2e82823          	sw	a4, -1232(a6)
    return *((volatile T*)(get_arg_addr(arg_idx)));
    1178:	000197b7          	lui	a5, 0x19
    117c:	40078793          	addi	a5, a5, 1024 # 19400 <substitutes.cpp.8dc51291+0x11ba3>
    1180:	0007ae83          	lw	t4, 0(a5)
    1184:	0047af83          	lw	t6, 4(a5)
    1188:	0087a803          	lw	a6, 8(a5)
    118c:	00c7af03          	lw	t5, 12(a5)
    uint32_t data = p_reg[0];
    1190:	ffb127b7          	lui	a5, 0xffb12
    1194:	1f07ae03          	lw	t3, 496(a5) # ffb121f0 <__global_pointer$+0x119f0>
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
    1198:	81c1a703          	lw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
    119c:	0001b8b7          	lui	a7, 0x1b
    local_mem_barrier = data;
    11a0:	81c1ac23          	sw	t3, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
    11a4:	1f87a303          	lw	t1, 504(a5)
    11a8:	00670613          	addi	a2, a4, 6
    11ac:	03200693          	li	a3, 50
    local_mem_barrier = data;
    11b0:	8061ac23          	sw	t1, -2024(gp) # ffb00018 <local_mem_barrier>
    11b4:	b3088793          	addi	a5, a7, -1232 # 1ab30 <substitutes.cpp.8dc51291+0x132d3>
    11b8:	18c6e063          	bltu	a3, a2, 1338 <main+0x898>
	    buffer[wIndex+TIMER_ID] = timer_id;
    11bc:	00271693          	slli	a3, a4, 0x2
    11c0:	00d78633          	add	a2, a5, a3
    11c4:	00500293          	li	t0, 5
    11c8:	00562023          	sw	t0, 0(a2)
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    11cc:	00468613          	addi	a2, a3, 4
    11d0:	00c78633          	add	a2, a5, a2
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    11d4:	00868693          	addi	a3, a3, 8
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    11d8:	01c62023          	sw	t3, 0(a2)
            wIndex += TIMER_DATA_UINT32_SIZE;
    11dc:	00370713          	addi	a4, a4, 3
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    11e0:	00d787b3          	add	a5, a5, a3
    11e4:	0067a023          	sw	t1, 0(a5)
            wIndex += TIMER_DATA_UINT32_SIZE;
    11e8:	80e1ae23          	sw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
            buffer [BUFFER_END_INDEX] = wIndex;
    11ec:	b2e8a823          	sw	a4, -1232(a7)
std::uint64_t get_noc_addr(std::uint32_t noc_x, std::uint32_t noc_y, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */
    return NOC_XY_ADDR(NOC_X(noc_x), NOC_Y(noc_y), addr);
    11f0:	00681813          	slli	a6, a6, 0x6
    11f4:	01f86833          	or	a6, a6, t6
    11f8:	0001b7b7          	lui	a5, 0x1b
  *ptr = val;
    11fc:	00002e37          	lui	t3, 0x2
    1200:	00001fb7          	lui	t6, 0x1
    1204:	00000893          	li	a7, 0
    std::uint16_t tiles_acked = tiles_acked_ptr[0];
    1208:	ffb58337          	lui	t1, 0xffb58
    uint32_t data = p_reg[0];
    120c:	ffb122b7          	lui	t0, 0xffb12
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
    1210:	03200393          	li	t2, 50
    1214:	b3078793          	addi	a5, a5, -1232 # 1ab30 <substitutes.cpp.8dc51291+0x132d3>
	    buffer[wIndex+TIMER_ID] = timer_id;
    1218:	00700993          	li	s3, 7
  return *ptr;
    121c:	ffb21737          	lui	a4, 0xffb21
  *ptr = val;
    1220:	092e0e13          	addi	t3, t3, 146 # 2092 <brisc.cc.d630dca8+0x420>
    1224:	800f8f93          	addi	t6, t6, -2048 # 800 <__firmware_stack_size+0x400>
    1228:	00100093          	li	ra, 1
  return *ptr;
    122c:	ffb20a37          	lui	s4, 0xffb20
    1230:	00b89a93          	slli	s5, a7, 0xb
    1234:	01da8ab3          	add	s5, s5, t4
    // kernel_profiler::mark_time(13);
    // kernel_profiler::mark_time(14);
    // kernel_profiler::mark_time(15);
    // kernel_profiler::mark_time(16);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
    1238:	131f0063          	beq	t5, a7, 1358 <main+0x8b8>
    123c:	00c32603          	lw	a2, 12(t1) # ffb5800c <__global_pointer$+0x5780c>
    1240:	01061613          	slli	a2, a2, 0x10
    1244:	01065613          	srli	a2, a2, 0x10
    1248:	01032683          	lw	a3, 16(t1)
    local_mem_barrier = data;
    124c:	80d1ac23          	sw	a3, -2024(gp) # ffb00018 <local_mem_barrier>
    } while (num_tiles_recv < num_tiles_u);
    1250:	01069693          	slli	a3, a3, 0x10
    1254:	0106d693          	srli	a3, a3, 0x10
    1258:	fed608e3          	beq	a2, a3, 1248 <main+0x7a8>
    uint32_t data = p_reg[0];
    125c:	1f02ac83          	lw	s9, 496(t0) # ffb121f0 <__global_pointer$+0x119f0>
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
    1260:	81c1a603          	lw	a2, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
    std::uint32_t rd_ptr_bytes = cb_read_interface[output].fifo_rd_ptr << 4;
    1264:	10892b03          	lw	s6, 264(s2)
    local_mem_barrier = data;
    1268:	8191ac23          	sw	s9, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
    126c:	1f82ac03          	lw	s8, 504(t0)
    1270:	00660693          	addi	a3, a2, 6
    1274:	004b1b13          	slli	s6, s6, 0x4
    local_mem_barrier = data;
    1278:	8181ac23          	sw	s8, -2024(gp) # ffb00018 <local_mem_barrier>
    127c:	0cd3e663          	bltu	t2, a3, 1348 <main+0x8a8>
	    buffer[wIndex+TIMER_ID] = timer_id;
    1280:	00261693          	slli	a3, a2, 0x2
    1284:	00d78bb3          	add	s7, a5, a3
    1288:	013ba023          	sw	s3, 0(s7)
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    128c:	00468b93          	addi	s7, a3, 4
    1290:	01778bb3          	add	s7, a5, s7
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    1294:	00868693          	addi	a3, a3, 8
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    1298:	019ba023          	sw	s9, 0(s7)
            wIndex += TIMER_DATA_UINT32_SIZE;
    129c:	00360613          	addi	a2, a2, 3
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    12a0:	00d786b3          	add	a3, a5, a3
    12a4:	0186a023          	sw	s8, 0(a3)
            wIndex += TIMER_DATA_UINT32_SIZE;
    12a8:	80c1ae23          	sw	a2, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
            buffer [BUFFER_END_INDEX] = wIndex;
    12ac:	00c7a023          	sw	a2, 0(a5)
  return *ptr;
    12b0:	82872603          	lw	a2, -2008(a4) # ffb20828 <__global_pointer$+0x20028>
    12b4:	82870693          	addi	a3, a4, -2008
  // kernel_profiler::mark_time(12);

  // kernel_profiler::mark_time(13);
  // kernel_profiler::mark_time(14);

  while (!ncrisc_noc_fast_write_ok(noc, cmd_buf));
    12b8:	fe061ce3          	bnez	a2, 12b0 <main+0x810>
  *ptr = val;
    12bc:	81c72e23          	sw	t3, -2020(a4)
    12c0:	81672023          	sw	s6, -2048(a4)
    12c4:	81572623          	sw	s5, -2036(a4)
    12c8:	81072823          	sw	a6, -2032(a4)
    12cc:	83f72023          	sw	t6, -2016(a4)
    12d0:	0016a023          	sw	ra, 0(a3)
    noc_nonposted_writes_num_issued[noc] += 1;
    12d4:	0005a683          	lw	a3, 0(a1)
    12d8:	00168693          	addi	a3, a3, 1
    12dc:	00d5a023          	sw	a3, 0(a1)
    noc_nonposted_writes_acked[noc] += num_dests;
    12e0:	00052683          	lw	a3, 0(a0)
    12e4:	00168693          	addi	a3, a3, 1
    12e8:	00d52023          	sw	a3, 0(a0)
  return *ptr;
    12ec:	204a2603          	lw	a2, 516(s4) # ffb20204 <__global_pointer$+0x1fa04>
 *
 * Return value: None
 */
FORCE_INLINE
void noc_async_write_barrier()  {
    while (!ncrisc_noc_nonposted_writes_flushed(loading_noc));
    12f0:	fec69ee3          	bne	a3, a2, 12ec <main+0x84c>
    tiles_acked_ptr[0] += num_tiles;
    12f4:	00c32683          	lw	a3, 12(t1)
    12f8:	00168693          	addi	a3, a3, 1
    12fc:	00d32623          	sw	a3, 12(t1)
    cb_read_interface[output].fifo_rd_ptr += num_words;
    1300:	10892683          	lw	a3, 264(s2)
    if (cb_read_interface[output].fifo_rd_ptr > cb_read_interface[output].fifo_limit) {
    1304:	10492603          	lw	a2, 260(s2)
    cb_read_interface[output].fifo_rd_ptr += num_words;
    1308:	08068693          	addi	a3, a3, 128
    130c:	10d92423          	sw	a3, 264(s2)
    if (cb_read_interface[output].fifo_rd_ptr > cb_read_interface[output].fifo_limit) {
    1310:	00d67863          	bgeu	a2, a3, 1320 <main+0x880>
        cb_read_interface[output].fifo_rd_ptr -= cb_read_interface[output].fifo_size;
    1314:	10092603          	lw	a2, 256(s2)
    1318:	40c686b3          	sub	a3, a3, a2
    131c:	10d92423          	sw	a3, 264(s2)
    1320:	00188893          	addi	a7, a7, 1
    1324:	f0dff06f          	j	1230 <main+0x790>
	} else {
            buffer [DROPPED_MARKER_COUNTER]++;
    1328:	0047a703          	lw	a4, 4(a5)
    132c:	00170713          	addi	a4, a4, 1
    1330:	00e7a223          	sw	a4, 4(a5)
    1334:	e45ff06f          	j	1178 <main+0x6d8>
    1338:	0047a703          	lw	a4, 4(a5)
    133c:	00170713          	addi	a4, a4, 1
    1340:	00e7a223          	sw	a4, 4(a5)
    1344:	eadff06f          	j	11f0 <main+0x750>
    1348:	0047a683          	lw	a3, 4(a5)
    134c:	00168693          	addi	a3, a3, 1
    1350:	00d7a223          	sw	a3, 4(a5)
        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, src_local_l1_addr, dst_noc_addr, size,
    1354:	f5dff06f          	j	12b0 <main+0x810>
    uint32_t data = p_reg[0];
    1358:	ffb127b7          	lui	a5, 0xffb12
    135c:	1f07a803          	lw	a6, 496(a5) # ffb121f0 <__global_pointer$+0x119f0>
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
    1360:	81c1a703          	lw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
    1364:	0001b5b7          	lui	a1, 0x1b
    local_mem_barrier = data;
    1368:	8101ac23          	sw	a6, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
    136c:	1f87a503          	lw	a0, 504(a5)
    1370:	00670613          	addi	a2, a4, 6
    1374:	03200693          	li	a3, 50
    local_mem_barrier = data;
    1378:	80a1ac23          	sw	a0, -2024(gp) # ffb00018 <local_mem_barrier>
    137c:	b3058793          	addi	a5, a1, -1232 # 1ab30 <substitutes.cpp.8dc51291+0x132d3>
    1380:	12c6e663          	bltu	a3, a2, 14ac <main+0xa0c>
	    buffer[wIndex+TIMER_ID] = timer_id;
    1384:	00271693          	slli	a3, a4, 0x2
    1388:	00d78633          	add	a2, a5, a3
    138c:	00600893          	li	a7, 6
    1390:	01162023          	sw	a7, 0(a2)
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    1394:	00468613          	addi	a2, a3, 4
    1398:	00c78633          	add	a2, a5, a2
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    139c:	00868693          	addi	a3, a3, 8
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    13a0:	01062023          	sw	a6, 0(a2)
            wIndex += TIMER_DATA_UINT32_SIZE;
    13a4:	00370713          	addi	a4, a4, 3
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    13a8:	00d787b3          	add	a5, a5, a3
    13ac:	00a7a023          	sw	a0, 0(a5)
            wIndex += TIMER_DATA_UINT32_SIZE;
    13b0:	80e1ae23          	sw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
            buffer [BUFFER_END_INDEX] = wIndex;
    13b4:	b2e5a823          	sw	a4, -1232(a1)
    uint32_t data = p_reg[0];
    13b8:	ffb127b7          	lui	a5, 0xffb12
    13bc:	1f07a803          	lw	a6, 496(a5) # ffb121f0 <__global_pointer$+0x119f0>
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
    13c0:	81c1a703          	lw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
    13c4:	0001b5b7          	lui	a1, 0x1b
    local_mem_barrier = data;
    13c8:	8101ac23          	sw	a6, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
    13cc:	1f87a503          	lw	a0, 504(a5)
    13d0:	00670613          	addi	a2, a4, 6
    13d4:	03200693          	li	a3, 50
    local_mem_barrier = data;
    13d8:	80a1ac23          	sw	a0, -2024(gp) # ffb00018 <local_mem_barrier>
    13dc:	b3058793          	addi	a5, a1, -1232 # 1ab30 <substitutes.cpp.8dc51291+0x132d3>
    13e0:	0cc6ee63          	bltu	a3, a2, 14bc <main+0xa1c>
	    buffer[wIndex+TIMER_ID] = timer_id;
    13e4:	00271693          	slli	a3, a4, 0x2
    13e8:	00d78633          	add	a2, a5, a3
    13ec:	00300893          	li	a7, 3
    13f0:	01162023          	sw	a7, 0(a2)
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    13f4:	00468613          	addi	a2, a3, 4
    13f8:	00c78633          	add	a2, a5, a2
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    13fc:	00868693          	addi	a3, a3, 8
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    1400:	01062023          	sw	a6, 0(a2)
            wIndex += TIMER_DATA_UINT32_SIZE;
    1404:	00370713          	addi	a4, a4, 3
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    1408:	00d787b3          	add	a5, a5, a3
    140c:	00a7a023          	sw	a0, 0(a5)
            wIndex += TIMER_DATA_UINT32_SIZE;
    1410:	80e1ae23          	sw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
            buffer [BUFFER_END_INDEX] = wIndex;
    1414:	b2e5a823          	sw	a4, -1232(a1)
    1418:	000197b7          	lui	a5, 0x19
    141c:	0047a783          	lw	a5, 4(a5) # 19004 <substitutes.cpp.8dc51291+0x117a7>
    1420:	02078a63          	beqz	a5, 1454 <main+0x9b4>
    1424:	00100793          	li	a5, 1
    1428:	00c02703          	lw	a4, 12(zero) # c <__firmware_stack_size-0x3f4>
    142c:	fef71ee3          	bne	a4, a5, 1428 <main+0x988>
    1430:	01002703          	lw	a4, 16(zero) # 10 <__firmware_stack_size-0x3f0>
    1434:	fef71ae3          	bne	a4, a5, 1428 <main+0x988>
    1438:	01402703          	lw	a4, 20(zero) # 14 <__firmware_stack_size-0x3ec>
    143c:	fef716e3          	bne	a4, a5, 1428 <main+0x988>
  return ptr[0];
    1440:	ffb12737          	lui	a4, 0xffb12
    1444:	1b072783          	lw	a5, 432(a4) # ffb121b0 <__global_pointer$+0x119b0>
  WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 | trisc_reset_mask);
    1448:	000076b7          	lui	a3, 0x7
    144c:	00d7e7b3          	or	a5, a5, a3
  ptr[0] = val;
    1450:	1af72823          	sw	a5, 432(a4)
    1454:	00802703          	lw	a4, 8(zero) # 8 <__firmware_stack_size-0x3f8>
    1458:	deeeb7b7          	lui	a5, 0xdeeeb
    145c:	aad78793          	addi	a5, a5, -1363 # deeeaaad <__global_pointer$+0xdf3ea2ad>
    1460:	00f70663          	beq	a4, a5, 146c <main+0x9cc>
    1464:	00100793          	li	a5, 1
    1468:	00f02423          	sw	a5, 8(zero) # 8 <__firmware_stack_size-0x3f8>
    146c:	08002a23          	sw	zero, 148(zero) # 94 <__firmware_stack_size-0x36c>
    uint32_t data = p_reg[0];
    1470:	ffb127b7          	lui	a5, 0xffb12
    1474:	1f07a803          	lw	a6, 496(a5) # ffb121f0 <__global_pointer$+0x119f0>
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
    1478:	81c1a703          	lw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
    147c:	0001b5b7          	lui	a1, 0x1b
    local_mem_barrier = data;
    1480:	8101ac23          	sw	a6, -2024(gp) # ffb00018 <local_mem_barrier>
    uint32_t data = p_reg[0];
    1484:	1f87a503          	lw	a0, 504(a5)
    1488:	fcf70613          	addi	a2, a4, -49
    148c:	fc800793          	li	a5, -56
    local_mem_barrier = data;
    1490:	80a1ac23          	sw	a0, -2024(gp) # ffb00018 <local_mem_barrier>
    1494:	b3058693          	addi	a3, a1, -1232 # 1ab30 <substitutes.cpp.8dc51291+0x132d3>
    1498:	02c7ea63          	bltu	a5, a2, 14cc <main+0xa2c>
            buffer [DROPPED_MARKER_COUNTER]++;
    149c:	0046a783          	lw	a5, 4(a3) # 7004 <noc.c.715ded3a+0x274>
    14a0:	00178793          	addi	a5, a5, 1
    14a4:	00f6a223          	sw	a5, 4(a3)
    14a8:	0000006f          	j	14a8 <main+0xa08>
    14ac:	0047a703          	lw	a4, 4(a5)
    14b0:	00170713          	addi	a4, a4, 1
    14b4:	00e7a223          	sw	a4, 4(a5)
    14b8:	f01ff06f          	j	13b8 <main+0x918>
    14bc:	0047a703          	lw	a4, 4(a5)
    14c0:	00170713          	addi	a4, a4, 1
    14c4:	00e7a223          	sw	a4, 4(a5)
    14c8:	f51ff06f          	j	1418 <main+0x978>
	    buffer[wIndex+TIMER_ID] = timer_id;
    14cc:	00271793          	slli	a5, a4, 0x2
    14d0:	00d78633          	add	a2, a5, a3
    14d4:	00400893          	li	a7, 4
    14d8:	01162023          	sw	a7, 0(a2)
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    14dc:	00478613          	addi	a2, a5, 4
    14e0:	00d60633          	add	a2, a2, a3
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    14e4:	00878793          	addi	a5, a5, 8
	    buffer[wIndex+TIMER_VAL_L] = time_L;
    14e8:	01062023          	sw	a6, 0(a2)
            wIndex += TIMER_DATA_UINT32_SIZE;
    14ec:	00370713          	addi	a4, a4, 3
	    buffer[wIndex+TIMER_VAL_H] = time_H;
    14f0:	00d787b3          	add	a5, a5, a3
    14f4:	00a7a023          	sw	a0, 0(a5)
            wIndex += TIMER_DATA_UINT32_SIZE;
    14f8:	80e1ae23          	sw	a4, -2020(gp) # ffb0001c <_ZN15kernel_profiler6wIndexE>
            buffer [BUFFER_END_INDEX] = wIndex;
    14fc:	b2e5a823          	sw	a4, -1232(a1)
    1500:	fa9ff06f          	j	14a8 <main+0xa08>
	...

00001510 <_Z15noc_set_cfg_regmm>:
    1510:	ffb007b7          	lui	a5, 0xffb00
    1514:	0007a783          	lw	a5, 0(a5) # ffb00000 <__global_pointer$+0xfffff800>
    1518:	3fec8737          	lui	a4, 0x3fec8
    151c:	04070713          	addi	a4, a4, 64 # 3fec8040 <substitutes.cpp.8dc51291+0x3fec07e3>
    1520:	00e79793          	slli	a5, a5, 0xe
    1524:	00e787b3          	add	a5, a5, a4
    1528:	00a787b3          	add	a5, a5, a0
    152c:	00279793          	slli	a5, a5, 0x2
    1530:	00b7a023          	sw	a1, 0(a5)
    1534:	00008067          	ret

00001538 <_Z15noc_get_cfg_regm>:
    1538:	ffb007b7          	lui	a5, 0xffb00
    153c:	0007a783          	lw	a5, 0(a5) # ffb00000 <__global_pointer$+0xfffff800>
    1540:	3fec8737          	lui	a4, 0x3fec8
    1544:	04070713          	addi	a4, a4, 64 # 3fec8040 <substitutes.cpp.8dc51291+0x3fec07e3>
    1548:	00e79793          	slli	a5, a5, 0xe
    154c:	00e787b3          	add	a5, a5, a4
    1550:	00a787b3          	add	a5, a5, a0
    1554:	00279793          	slli	a5, a5, 0x2
    1558:	0007a503          	lw	a0, 0(a5)
    155c:	00008067          	ret

00001560 <exit>:
    1560:	0000006f          	j	1560 <exit>
    1564:	00000013          	nop
    1568:	00000013          	nop
    156c:	00000013          	nop

00001570 <wzerorange>:
    1570:	00b50863          	beq	a0, a1, 1580 <wzerorange+0x10>
    1574:	00052023          	sw	zero, 0(a0)
    1578:	00450513          	addi	a0, a0, 4
    157c:	ff5ff06f          	j	1570 <wzerorange>
    1580:	00008067          	ret
	...
