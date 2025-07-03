    asm volatile("sfpstore 0,%0,6,0" : : "x"(failed.get()));

    uint32_t ary[8];
    dbg_read_dest_acc_row(0, ary);
    auto* args = reinterpret_cast<tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
    args[0] = ary[0] & 0xffff;
