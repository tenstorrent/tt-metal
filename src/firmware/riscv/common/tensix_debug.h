#define PERF_CNT_INSTR_THREAD_DAISY_ID 0
#define PERF_CNT_TDMA_DAISY_ID 1

// perf_cnt IDs
#define PERF_CNT_INSTR_THREAD 0
#define PERF_CNT_INSTR_ISSUE  1

// Debug daisy IDs
#define DEBUG_DAISY_INSTRN_THREAD 1
#define DEBUG_DAISY_INSTRN_ISSUE 2
#define DEBUG_DAISY_L0 7
#define DEBUG_DAISY_L1 8

// Registers
// Debug daisy registers are kernel regs
#define DEBUG_MUX_CTRL           50
#define DEBUG_MUX_RD             51
#define DEBUG_MUX_CTRL_SigSel   15:0
#define DEBUG_MUX_CTRL_DaisySel 7:0
#define DEBUG_MUX_CTRL_RdSel    3:0
#define DEBUG_MUX_CTRL_En       0:0
#define DEBUG_MUX_RD_RdVal      31:0

// Local risc registers for everything else
#define RISCV_DBG_REGS_START_ADDR            0xFFB12000
#define RISCV_DBG_PERF_CNT_INSTRN_THREAD0    0xFFB12000
#define RISCV_DBG_PERF_CNT_INSTRN_THREAD1    0xFFB12004
#define RISCV_DBG_PERF_CNT_INSTRN_THREAD2    0xFFB12008
#define RISCV_DBG_PERF_CNT_TDMA0             0xFFB1200C
#define RISCV_DBG_PERF_CNT_TDMA1             0xFFB12010
#define RISCV_DBG_PERF_CNT_TDMA2             0xFFB12014
#define RISCV_DBG_PERF_CNT_FPU0              0xFFB12018
#define RISCV_DBG_PERF_CNT_FPU1              0xFFB1201C
#define RISCV_DBG_PERF_CNT_FPU2              0xFFB12020
#define RISCV_DBG_PERF_CNT_L0_0              0xFFB12024
#define RISCV_DBG_PERF_CNT_L0_1              0xFFB12028
#define RISCV_DBG_PERF_CNT_L0_2              0xFFB1202C
#define RISCV_DBG_PERF_CNT_L1_0              0xFFB12030
#define RISCV_DBG_PERF_CNT_L1_1              0xFFB12034
#define RISCV_DBG_PERF_CNT_L1_2              0xFFB12038
#define RISCV_DBG_PERF_CNT_ALL               0xFFB1203C
#define DBG_L0_MEM_REG0                      0xFFB12040
#define DBG_L0_MEM_REG1                      0xFFB12044
#define DBG_L1_MEM_REG0                      0xFFB12048
#define DBG_L1_MEM_REG1                      0xFFB1204C
#define DBG_L1_MEM_REG2                      0xFFB12050
#define DBG_BUS_CNTL_REG                     0xFFB12054
#define WALL_CLOCK_0                         0xFFB121F0
#define WALL_CLOCK_1                         0xFFB121F4
#define DBG_L0_READBACK_OFFSET               0xFFB12200
#define DBG_L1_READBACK_OFFSET               0xFFB12280
#define RISCV_DBG_ARRAY_RD_EN                0xFFB12060
#define RISCV_DBG_ARRAY_RD_CMD               0xFFB12064
#define RISCV_DBG_FEATURE_DISABLE            0xFFB12068
#define RISCV_CG_REGBLOCKS_CTRL              0xFFB12070
#define RISCV_CG_FPU_CTRL                    0xFFB12074
#define RISCV_THREAD0_CREG_RDDATA            0xFFB12078
#define RISCV_THREAD1_CREG_RDDATA            0xFFB1207C


// Useful defines for perf counters and debug bus
#define PERF_CNT_MODE_FREE 0
#define PERF_CNT_MODE_STOP 1
#define PERF_CNT_MODE_WRAP 2

#define DEBUG_MEM_MODE_MANUAL_WR 0
#define DEBUG_MEM_MODE_AUTO_WR   1
#define DEBUG_MEM_MODE_MANUAL_RD 2
#define DEBUG_MEM_MODE_AUTO_RD   3

// each perf cnt has an associated daisy ID
const uint perf_cnt_daisy_id[] = { DEBUG_DAISY_INSTRN_THREAD, DEBUG_DAISY_INSTRN_ISSUE, 8, 0};

inline uint wait(int cycles){
  int count = 0;
  uint bla = 0;

  volatile uint * mailbox    = (volatile uint *)0xFFB00000;
  while (count < cycles) {
      bla = mailbox[0];
      count++;
  }
  return bla;
}

inline void dbg_daisy_disable(
    bool use_setc, 
      volatile uint * instrn_buf ,
        uint & debug_sel_reg
)
{
    if(use_setc) {
        debug_sel_reg = debug_sel_reg & 0xefffffff;
        //ex_setc(DEBUG_MUX_CTRL, debug_sel_reg, instrn_buf);
    } else {
        uint reg_value;
        debug_sel_reg = debug_sel_reg & 0xcfffffff;
        reg_value = memory_read(DBG_BUS_CNTL_REG);
        memory_write(DBG_BUS_CNTL_REG, (reg_value & 0xfcffffff) | debug_sel_reg);
    }
}
inline void dbg_daisy_enable(
    bool use_setc, 
      volatile uint * instrn_buf ,
        uint &debug_sel_reg
)
{
    if(use_setc) {
        debug_sel_reg = debug_sel_reg | 0x10000000;
        //ex_setc(DEBUG_MUX_CTRL, debug_sel_reg, instrn_buf);
    } else {
        uint reg_value;
        debug_sel_reg = debug_sel_reg | 0x30000000;
        reg_value = memory_read(DBG_BUS_CNTL_REG);
        memory_write(DBG_BUS_CNTL_REG, (reg_value & 0xfcffffff) | debug_sel_reg);
    }
}
inline void dbg_stop_perf_cnt(
      volatile uint * instrn_buf ,
        uint    reg_sel
)
{
    memory_write(reg_sel, 0x00000002);
    memory_write(reg_sel, 0x00000000);
}
inline void dbg_start_perf_cnt(
      volatile uint * instrn_buf ,
        uint    reg_sel
)
{
    memory_write(reg_sel, 0x00000001);
    memory_write(reg_sel, 0x00000000);
}
inline void dbg_mem_write(
      volatile uint * instrn_buf ,
        uint    reg
)
{
    uint reg_value;
    reg_value = memory_read(reg);
    reg_value = reg_value | 0x00001000;
    memory_write(reg, reg_value);
    reg_value = reg_value & 0xffffefff;
    memory_write(reg, reg_value);
}

// Note - need to run properly set up dbg_mem_read before this instruction
// This instruction only reads the local reg, assuming data has already been read from L0 and propagated
// This is a hack wihtout a proper 128-bit data structure, so just have to call it 4 times
inline uint dbg_l0_data_read(
    char index
)
{
    uint reg_value;
    reg_value = memory_read(DBG_L0_READBACK_OFFSET+index);
    return reg_value;
}
inline uint dbg_l1_data_read(
    char index
)
{
    uint reg_value;
    reg_value = memory_read(DBG_L1_READBACK_OFFSET+index);
    return reg_value;
}

inline uint get_wall_clock(
    char index
)
{
    uint reg_value;
    reg_value = memory_read(WALL_CLOCK_0+(4*index));
    return reg_value;
}

inline void dbg_mem_read(
      volatile uint * instrn_buf ,
        uint    reg
)
{
    uint reg_value;
    reg_value = memory_read(reg);
    reg_value = reg_value | 0x00002000;
    memory_write(reg, reg_value);
    reg_value = reg_value & 0xffffdfff;
    memory_write(reg, reg_value);
}
inline void dbg_enable_mem_dump_l1(
        uint    reg,
        uint start_addr,
        uint end_addr,
        char mode
)
{

    uint reg_value = start_addr;
    memory_write(reg, reg_value);

    reg=reg+4; 
    reg_value  = end_addr;
    memory_write(reg, reg_value);

    reg=reg+4; 
    reg_value  = mode;
    memory_write(reg, reg_value);
//value = (value & ~mask) | (newvalue & mask);}
}
inline void dbg_enable_mem_dump_l0(
      volatile uint * instrn_buf ,
        uint    reg,
        uint start_addr,
        uint end_addr,
        char mode
)
{

    uint reg_value = end_addr;
    short lower_word = start_addr;
    reg_value = reg_value << 16 & 0xffff0000;
    reg_value = reg_value | lower_word;
    memory_write(reg, reg_value);

    reg=reg+4; 
    reg_value  = mode;
    memory_write(reg, reg_value);
//value = (value & ~mask) | (newvalue & mask);}
}
inline void dbg_set_perf_cnt_params(
      volatile uint * instrn_buf ,
        uint    reg_sel,
        uint ref_period,
        uint mode
)
{

    memory_write(RISCV_DBG_PERF_CNT_INSTRN_THREAD0, ref_period);
    memory_write(RISCV_DBG_PERF_CNT_INSTRN_THREAD0+4, mode);
}
inline void dbg_sig_sel(
    bool use_setc, 
      volatile uint * instrn_buf ,
        uint & debug_sel_reg,
        uint    sig_sel
)
{
    uint sel_to_write;
    if(use_setc) {
        sel_to_write = (sig_sel) & 0x0000ffff;
        debug_sel_reg = (debug_sel_reg & 0xffff0000) | sel_to_write;
        //ex_setc(DEBUG_MUX_CTRL, debug_sel_reg, instrn_buf);
    } else {
        uint reg_value;
        sel_to_write = (sig_sel) & 0x0000ffff;
        reg_value = memory_read(DBG_BUS_CNTL_REG);
        debug_sel_reg = (reg_value & 0xffff0000) | sel_to_write;
        memory_write(DBG_BUS_CNTL_REG, debug_sel_reg);
    }
}
inline void dbg_daisy_sel(
    bool use_setc, 
      volatile uint * instrn_buf ,
        uint & debug_sel_reg,
        uint    daisy_sel
)
{
    uint sel_to_write;
    if(use_setc) {
        sel_to_write = (daisy_sel <<16) & 0x00ff0000;
        debug_sel_reg = (debug_sel_reg & 0xff00ffff) | sel_to_write;
        //ex_setc(DEBUG_MUX_CTRL, debug_sel_reg, instrn_buf);
    } else {
        uint reg_value;
        sel_to_write = (daisy_sel <<16) & 0x00ff0000;
        reg_value = memory_read(DBG_BUS_CNTL_REG);
        debug_sel_reg = (reg_value & 0xff00ffff) | sel_to_write;
        memory_write(DBG_BUS_CNTL_REG, debug_sel_reg);
    }
}
inline void dbg_soft_reset(
    bool use_read,
    uint reset_reg
)
{
    if (use_read) {
        uint reset_status;
        reset_status = memory_read(RISCV_DEBUG_REG_SOFT_RESET_0);
        memory_write(RISCV_DEBUG_REG_SOFT_RESET_0, (reset_status|reset_reg));
        reset_status = memory_read(RISCV_DEBUG_REG_SOFT_RESET_0);
        memory_write(RISCV_DEBUG_REG_SOFT_RESET_0, (reset_status&~reset_reg));
    } else {
        memory_write(RISCV_DEBUG_REG_SOFT_RESET_0, reset_reg);
        memory_write(RISCV_DEBUG_REG_SOFT_RESET_0, 0);
    }
}
