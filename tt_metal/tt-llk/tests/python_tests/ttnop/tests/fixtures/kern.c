// A freestanding RV32 program with rich control flow (recursion, a loop, nested
// calls, a switch, volatile stores) so that ttnop's detours -- verbatim copies,
// jal recomputation, branch inversion, and call return-trampolines -- are all
// exercised when we instrument it. It prints a single deterministic checksum so
// the test can assert original-vs-instrumented equivalence.
//
// Built with -nostdlib and a Tensix-like linker script (.text in its own
// segment, ELF header outside it), then run under qemu-riscv32.

static long sys(long n, long a, long b, long c)
{
    register long a7 asm("a7") = n, a0 asm("a0") = a, a1 asm("a1") = b, a2 asm("a2") = c;
    asm volatile("ecall" : "+r"(a0) : "r"(a7), "r"(a1), "r"(a2) : "memory");
    return a0;
}

static void wr(const char* s, long n)
{
    sys(64, 1, (long)s, n);
} // write(1,..)

volatile int sink; // forces real stores that survive optimization

static int fib(int n)
{ // recursion -> jal calls, branches
    if (n < 2)
    {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

static int classify(int x)
{ // switch -> jump/branch lowering
    switch (x & 7)
    {
        case 0:
            return 11;
        case 1:
            return 22;
        case 2:
            return 33;
        case 3:
            return 44;
        case 4:
            return 55;
        case 5:
            return 66;
        case 6:
            return 77;
        default:
            return 88;
    }
}

static int kernel(int n)
{
    int acc = 0;
    for (int i = 0; i < n; i++)
    { // loop -> branches
        if (i & 1)
        {
            acc += classify(i); // conditional -> branch + call
        }
        else
        {
            acc -= fib(i % 12); // call + modulo
        }
        sink = acc; // store every iteration
    }
    return acc + fib(15);
}

static int put_int(char* buf, int v)
{
    char tmp[16];
    int neg = v < 0, k = 0;
    unsigned uv = neg ? -(unsigned)v : (unsigned)v;
    if (!uv)
    {
        tmp[k++] = '0';
    }
    while (uv)
    {
        tmp[k++] = '0' + uv % 10;
        uv /= 10;
    }
    int j = 0;
    if (neg)
    {
        buf[j++] = '-';
    }
    while (k)
    {
        buf[j++] = tmp[--k];
    }
    buf[j++] = '\n';
    return j;
}

void _start(void)
{
    char buf[32];
    int r = kernel(200);
    wr(buf, put_int(buf, r));
    sys(93, 0, 0, 0); // exit(0)
}
