.. _llk:

Low Level Kernels
*****************

Overview
========

SFPI is the programming interface to the SFPU.  It consists of a C++ wrapper
around a RISCV GCC compiler base which has been extended with vector data types and
``__builtin`` intrinsics to generate SFPU instructions.  The wrapper provides a
C++ like interface for programming.

Compiler Options/Flags
======================

The following flags must be specified to compile SFPI kernels:

.. code-block:: c++

  -mcpu=<cpu> -fno-exceptions

where ``cpu`` is one of:

  * tt-wh-tensix
  * tt-bh-tensix
  * tt-qsr32-tensix

Note that the arch specification above overrides any ``-march=<xyz>`` that comes after it on the command line.

Further, the following options disable parts of the SFPI enabled compiler:

  * ``-mno-tt-tensix-warn``: disable sfpu specific warnings/errors
  * ``-mno-tt-tensix-optimize-combine``: disable sfpu instruction combining
  * ``-mno-tt-tensix-optimize-cc``: disable sfpu CC optimizations
  * ``-mno-tt-tensix-optimize-replay``: disable sfpu REPLAY optimizations

Example
-------

Before going into details, below is a simple example of SFPI code showcasing the main capabilities of SFPI kernels (please refer to :ref:`Writing Custom SFPU Operations<custom_sfpu>` for details on writing and invoking them). The example itself is not particularly useful, but it does show a number of features of SFPI:

.. code-block:: c++

    void silly(bool take_abs)
    {
        // dst_reg[n] loads into a temporary LREG
        vFloat a = dst_reg[0] + 2.0F;

        // This emits a load, move, mad
        dst_reg[3] = a * -dst_reg[1] + vConstFloatPrgm0 + 0.5F;

        // This emits a load, loadi, mad (a * dst_reg[] goes down the mad path)
        dst_reg[4] = a * dst_reg[1] + 1.2F;

        // This emits two loadis and a mad
        dst_reg[4] = a * 1.5F + 1.2F;

        // This emits a loadi (into tmp), loadi (as a temp for 1.2F) and a mad
        vFloat tmp = sFloat16a(bitpattern);
        dst_reg[5] = a * tmp + 1.2F;

        v_if ((a >= 4.0F && a < 8.0F) || (a >= 12.0F && a < 16.0F)) {
            vInt b = exexp(a, ExponentMode::Biased);
            b &= 0xAA;
            v_if (b >= 130) {
                dst_reg[6] = setexp(a, 127);
            }
            v_endif;
        } v_elseif (a == 3.0f) {
            // RISCV branch
            if (take_abs) {
                dst_reg[7] = abs(a);
            } else {
                dst_reg[7] = a;
            }
        } v_else {
            vInt exp = lz(a) - 19;
            exp = ~exp;
            dst_reg[8] = -setexp(a, exp);
        }
        v_endif;
    }

The main things to note from the example are:

  * Constants are expressed as scalars but are expanded to the width of the vector
  * ``v_if`` (and related) predicate execution of vector operations such that only enabled vector elements are written
  * The compiler views ``v_if`` and ``v_elseif`` as straight-line code, ie, both sides of the conditionals are executed
  * RISCV conditional and looping instructions work as expected (only one side executed)
  * Math expressions for vectors work across all enabled vector elements
  * Presently, ``v_endif`` is required to close out all ``v_if``/``v_elseif``/``v_else`` chains

And also some implications

  * Standard C++ ``if`` statements cannot be used to handle vector conditionals
  * ``v_if`` implements condition via predication - only vector operations are predicated
  * Same performance consideration apply SFPI as for any SIMT architecture - avoid divergent ``v_if`` execution paths

Details
=======

Namespace
---------

All the data types/objects/etc. listed below fall within the ``sfpi``
namespace.

User Visible Data Types
-----------------------

The following vector data types are visible to the programmer:

  * ``vFloat`` - 32-bit IEEE float
  * ``vInt`` - 32-bit 2's complement integer
  * ``vUInt``- 32-bit unsigned integer
  * ``vSMag``- 32-bit sign-magnitude integer
  * ``vMag`` - known-positive 32-bit sign-magnitude integer
  * ``vBool`` - boolean result of a conditional or logical operator

The following restricted-range types are also available. These are
held in one of the above data representations, but hold values
restricted to a subdomain of the type.

  * ``vFloat16a`` - fp16a format (5-bit exponent, 10-bit fraction)
  * ``vFloat16b`` - fp16b format (8-bit exponent, 7-bit fraction)
  * ``vUInt16``- 16-bit unsigned integer
  * ``vUInt8``- 8-bit unsigned integer
  * ``vSMag16``- 16-bit sign-magnitude integer
  * ``vSMag8``- 8-bit sign-magnitude integer

Each of the ``v`` types is a strongly typed wrapper around the weakly
typed compiler data type ``__rvtt_vec_t``. All types have the same element
format, but possibly with a restricted range. The width of this type
depends on the target architecture. On Wormhole, Blackhole & Quasar this is
a vector of 32 32-bit values. Users should be aware that vector length
may change with future architectures.

User Visible Constants
^^^^^^^^^^^^^^^^^^^^^^

Constant registers are implemented as objects which can be referenced wherever a vector can be used. On Wormhole and Blackhole the following variables are defined:

  * ``vConstTileId``, counts by two through the vector elements: [0, 2, 4..62]
  * ``vConstFloatPrgm0``, ``vConstIntPrgm0``
  * ``vConstFloatPrgm1``, ``vConstIntPrgm1``
  * ``vConstFloatPrgm2``, ``vConstIntPrgm2``

Note: previously the vector constants ``1.0f``, ``0.0f``, ``-1.0f``
and ``0.8373f`` were also available as named constants. Just use the
floating literals (possibly converted to ``vFloat``), the compiler
knows what to do.

User Visible Objects
^^^^^^^^^^^^^^^^^^^^

 * ``dst_reg[]`` is an array used to access the destination register
 * ``l_reg[]`` is an array used to load/store to specific SFPU registers

LRegs are the SFPU's general purpose vector registers.  The ``LRegs``
enum enumerates these registers.

On Quasar, ``SrcS`` is accessed via user-visible types:
``UnpackSrcS``, ``PackSrcS`` and ``ComputeSrcS``. Declare a local
variable of the appropriate type, and then access just as ``dst_reg``
is accessed.

Macros
^^^^^^

The only macros used within the wrapper implement the predicated conditional processing mechanism. These (of course) do not fall within the SFPI namespace and for brevity run some chance of a namespace collision. They are:

  * ``v_if(COND)``
  * ``v_elseif(COND)``
  * ``v_else``
  * ``v_endif``
  * ``v_block``
  * ``v_endblock``
  * ``v_and(COND)``

The conditionals work mostly as expected but note the required ``v_endif`` at the end of an if/else chain. Forgetting this results in compilation errors as the ``v_if`` macro contains a ``{`` which is matched by the ``v_endif``:

.. code-block:: c++

    v_if (a < b) {
        dst_reg[0] = a;
    } v_elseif (a > b) {
        dst_reg[0] = b;
    } v_else {
        dst_reg[0] = a + b;
    }
    v_endif;

``dst_reg[0]`` is assigned ``a`` where ``a < b``, ``b`` where ``a > b`` and ``a + b`` where ``a == b``.

However note that ``v_if`` and alike works via predication. In other words, both sides of the conditional are executed and only the enabled vector elements are written. RISC-V instructions are executed normally. For example:

.. code-block:: c++

    v_if (a < b) {
        DPRINT("a < b\n");
    } v_else {
        dst_reg[0] = b;
        DPRINT("a >= b\n");
    }
    v_endif;

Will result in both ``a < b`` and ``a >= b`` being printed, but only the elements where ``a >= b`` being written to ``dst_reg[0]``.

``v_block`` and ``v_and`` allow for the following code to progressively "narrow" the CC state:

.. code-block:: c++

    v_block {
        for (int x = 0; x < n; x++) {
            v1 = v1 - 1;
            v_and (v1 >= 0);
            v2 *= 2;
        }
    }
    v_endblock;

``v_and`` can be used inside any predicated conditional block (i.e., a ``v_block`` or a ``v_if``).

Data Type Details
-----------------

Loading and Storing
^^^^^^^^^^^^^^^^^^^

Values may be transfered to and from ``dst_reg`` (and on Quasar, one
of the SrcS objects):

.. code-block:: c++

    vFloat a = dst_reg[0];
    vUInt b = dst_reg[1];

    dst_reg[0] = a;
    dst_reg[2].mode<DataLayout::FP16b>() = a;
    dst_reg[3].mode<DataLayout::SM32>(2) = b;

If no ``mode`` override is provided, the data representation in
``dst_reg`` depends on the type being transferred, and in some cases
tha Architecture. You may override that default with the ``mode``
function, which optionally specifies a data representation, and an
optional addr_mode operand. This may be specified on both loads and
stores.  The following data representations and defaults are
available:

  * FSrcB - (vFloat) dynamic float representation
  * F32 - 32-bit float
  * F16a - (vFloat16a) 16-bit float16a
  * F16b - (vFloat16b) 16-bit float16b
  * I32 - (vInt, except Wormhole), 32-bit 2's complement integer
  * U32 - (vUInt), 32-bit unsigned integer
  * U16 - (vUInt16), 16-bit unsigned integer
  * SM32 - (vSMag), 32-bit sign-magnitude integer
  * SM16 - (vSMag16), 16-bit sign-magnitude integer
  * M32 - (vMag), 32-bit magnitude only integer

On Wormhole, the default mode for ``vInt`` is ``SM32``. In all cases
when transfering a ``vInt`` to or from ``SM32``, or tranferring
``vSMag`` to or from ``I32`` a conversion operation is inserted -- on
Wormhole this is part of the load or store, on other architectures it
is a separate operation. It is unspecified how 2's complement's most
negative value converts to sign-magnitude.  Not all data
representations are permitted for all types.

Quasar's SrcS reg accessors may also use a `done` modifier, to set the
''done'' bit in the load or store:

.. code-block:: c++

    ComputeSrcS srcs;
    vFloat a = srcs[0];
    ...
    srcs[0].done() = a;

``done`` and ``mode`` may be combined in either order on a single access.

Conversion and Type Punning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Value-preserving conversions that are also bit-preserving are
supported implicitly (or may be specified explicitly):

  * ``vFloat16a`` and ``vFloat16b`` readily convert to ``vFloat``
  * ``vUInt16` converts to ``vUInt``
  * ``vSMag16`` converts to ``vSMag``
  * ``vMag`` converts to ``vInt``, ``vUInt`` and ``vSMag``.

Potentially value & bit preserving conversions may be specified using
a functions-style cast.  For instance:

.. code-block:: c++

    vUInt a = dst_reg[0];
    auto b = vInt (a);
    auto c = vUInt16 (a);

This conversion preves the bit pattern, but is not implicit (unlike
c++'s scalar ``int`` and ``unsigned`` types.

Other bit-preserving conversions may be explicitly specified with the `as``
function:

.. code-block:: c++

    vFloat a = dst_reg[0];
    auto b = as<vUInt> (a);

Other value-preserving (or approximating) conversions use the
``convert`` function:

.. code-block:: c++

    vFloat a = dst_reg[0];
    auto b = convert<vFloat16b> (a, RoundMode::Nearest);
    auto c = convert<vSMag> (a); // convert to sign-mag
    auto d = convert<vInt> (c); // convert smag->int

the ``RoundMode`` operand is optional, and the following are provided:

  * ``NearestAway`` - round to nearest, ties round away from zero (Wormhole & Blackhole)
  * ``NearestEven`` - round to nearest, ties round to even fraction (Quasar)
  * ``NearestStochastic`` - round to nearest, ties round stochastically (default)
  * ``Zero`` - round to zero (not Wormhole)
  * ``Nearest`` - Alias for ``NearestAway``, or ``NearestEven``

Operators
^^^^^^^^^

The ``vFloat``, ``vInt`` and ``vUInt`` types support infix ``+` &
``-`` operators. The unary operators are provided.  For binary
operators both operands can be the same, or related, vector types, or
second may be a scalar value of the vector element type.  The ``-``
operator also permits the first operand to be a scalar.  The
``vFloat`` type also provides ``*``, and the integer types provide
``&``, ``|`` and ``^``. You may not mix the signedness of integer
operands. For the integer operations, the second operand may also be a
``vMag`` type (the ``vMag`` type itself does not provide operators).

The modifying variants, ``OP=``, are available.

Conditional operators are provided -- ``==``, ``!=``, ``<``, ``>=``,
``>`` & ``<=``. These produce a ``vBool`` result, which may be used
directly or indirectly in a ``v_if`` conditional. Both operands must
be related vector types, or the second operand may be an appropriate
scalar operator, or, for integral comparisons, may be a ``vMag`` type.

``vBool``s may be combined with ``&&``, ``||`` and ``!``
operations. Note that these are not short-circuiting.

Note: There is currently a compiler defect regarding signed and
unsigned integer comparisons, where ordering comparisons are only
correct when the two operands are within 2^31 of eachother. Also,
floating point comparisons use the multiply-add unit, which means
comparisons are not strictly conforming -- specifically infinities and
signed zeroes behave differently.

Scalar Values
^^^^^^^^^^^^^

Scalar values may be converted to vectors by using the appropriate
constructor. In the general case this takes 2 instructions, to load
the low and high halves of a 32-bit value. You may use the
``sFloat16a`` and ``sFloat16b` types to pun a 16-bit integer to the
specified fp16 representation. You may also use `sFloat16b` to convert
a scalar ``float`` to fp16b format, ensuring a single load immediate
instruction is used.

The compiler optimizes constant loading, using known-constant
register values, operations on those known constants, or an optimized
sequence of high and/or low load immediates. If the value being loaded
is dynamic, it will take advantage of knowing the scalar type is
representable in 16 bits (``int16_t``, ``uint16_t``, or one of the
``sFloat16a`` or ``sFloat16b`` types).

Library
-------

The sfpi library also provides the following API. In many cases below,
``vFloat`` implies any float vector type, ``vSMag`` any sign-magnitude
vector type and ``vUInt`` any unsigned vector type.

.. code-block:: c++

    vInt exexp(vFloat v, ExponentMode = ExponentMode::Unbiased);

Extracts a biased or unbiased exponent as a 2's complement
integer. ``ExponentMode`` may be ``Unbiased`` or ``Biased``.

.. code-block:: c++

    vMag exman(vFloat v, MantissaMode = MantissaMode::FractionOnly);

Extracts the mantissa of v.  ``MantissaMode`` may be
``FractionOnly`` or ``WithUnitBit`` (also ``ImplicitOne``).

.. code-block:: c++

    vMag exsgn({vFloat,vInt,vSMag} v);

Extracts the sign bit

.. code-block:: c++

    vFloat setexp(vFloat v, int exp);
    vFloat setexp(vFloat v, vInt exp);
    vFloat copyexp(vFloat v, vFloat src);
    vFloat addexp(vFloat v, int delta);

Replaces the exponent of ``v`` with the value of ``exp``, or the
exponent bits of ``src``. ``addexp`` adjusts the exponent by adding ``delta``.

.. code-block:: c++

    vFloat setman(vFloat v, unsigned man);
    vFloat setman(vFloat v, {vUInt,vSMag} man);
    vFloat copyman(vFloat v, vFloat src);

Replaces the mantissa of  ``v`` with the value of ``man`` or the
mantissa bits of ``src``.

.. code-block:: c++

    {vFloat,vMag} setsgn({vFloat,vMag} v, int sgn);
    vMag setsgn(vUInt v, int sgn);
    {vFloat,vMag} setsgn2({vFloat,vMag} v, {vInt,vUInt,vSMag} sgn)
    {vFloat,vMag} copysgn({vFloat,vMag} v, {vFloat,vInt,vSMag} cpy)

Replaces the sign bit of ''v'' with the value of ``sgn``, or copies
that of ``cpy``.  Note: ``setsgn2`` will be renamed once the
deprecated ``setsgn`` function that matches its signature is deleted.

.. code-block:: c++

    vFloat abs(vFloat v);
    vMag abs({vInt|vSMag} v);

Returns the absolute value of ''v''.

.. code-block:: c++

    vInt lz({vInt,vUInt,vSMag} v, LZMode = LZMode::All)

Returns the count of leading (left-most) zeros of ''v''. ``LZMode``
may be ``All`` or ``IgnoreSign`` (treats bit 31 as zero).

.. code-block:: c++

    vUInt shft(vUInt v, int amt, ShiftMode = ShiftMode::Logical);
    vUInt shft(vUInt v, vInt amt, ShiftMode = ShiftMode::Logical);
    vInt shft(vInt v, int amt, ShiftMode = ShiftMode::Arithmetic);
    vInt shft(vInt v, vInt amt, ShiftMode = ShiftMode::Arithmetic);

Performs a left shift (when ''amt'' is positive) or right shift (when
''amt'' is negative) of ''v'' by ''amt'' bits. ``ShiftMode`` may be
``Logical`` or ``Arithmetic``.  Wormhole does not support arithmetic
shifts and a compilation error will occur unless one explicitly
specifies ``Logical``.

.. code-block:: c++

    vBool is_nan (vFloat v);
    vBool is_finite (vFloat v);
    vBool is_normal (vFloat v);
    vBool is_subnormal (vFloat v);
    vBool is_zero (vFloat v);
    vBool is_inf (vFloat v);
    vBool is_pos (vFloat v);
    vBool is_neg (vFloat v);

Compute the named feature of `v`. ``is_nbormal`` is true when ``v`` is
a finite non-zero, non-subnormal number. ``is_finite`` is true when
``v`` is neither a nan nor an infinity.

.. code-block:: c++

   vMag fractional_mul ({vFloat,vUInt,vSMag} a, {vFloat,vUInt,vSMag} b, FractionalHalf = FractionalHalf::Low);

compute 23 bits of product of the low (fractional) 23-bits of ``a``
and ``b``.  ``FractionalHalf`` may be either ``Low`` or ``High``.  Not
available on Wormhole.

.. code-block:: c++

    void swap(vType &a, vType &b);

Swaps the values of ``a`` and ``b``.  Note that this uses the
``sfpswap`` instruction, rather than simply exchanging registers
(unlike ``std::swap``).

.. code-block:: c++

    {vFloat,vSMag} min({vFloat,vSMag} a, {vFloat,vSmag} b);
    vFloat min(vFloat a, float b);
    {vFloat,vSMag} max({vFloat,vSMag} a, {vFloat,vSmag} b);
    vFloat max(vFloat a, float b);
    {vFloat,vSMag} clamp({vFloat,vSMag} a, {vFloat,vSmag} lower, {vFloat,vSmag} upper);
    vFloat clamp(vFloat a, float lower, float upper);
    vFloat symmetric_clamp(vFloat a, float bound);

Return the minimum, maximum or clamped value.  ``symmetric_clamp``
clamps to the range `[-bound,+bound]`.

.. code-block:: c++

    std::pair<{vFloat,vSMag},{vFloat,vSmag}> min_max ({vFloat,vSmag} a, {vFloat,vSmag} b, unsigned mask = 0);

Separate `a` & `b` elements into minima and maxima according to
``mask``. Mask is either a 32-bit combination of 0xff or 0x00 bytes,
or a 4-bit number.  Where each byte (or bit) is zero, the minimum
element will be placed in the first part of the paired result.  Where
it is non-zero, the maximum will be chosen.  Thus, by default this
returns the min/max pair.  With a ``mask`` of ``0xffffffff`` or ``0xf``
the max/min pair will be returned. Non-permitted ``mask`` values will
result in a compilation error. It may be convenient to use a
structured binding to hold the result:

.. code-block:: c++

    auto [min, max] = min_max (a, b);

.. code-block:: c++

    vInt rand ();

Return a random integer. Due to hardware limitations, the random
distribution is not flat. Not available on Wormhole.

.. code-block:: c++

    vFloat rectified_linear_unit (vFloat src);

Compute ReLU, which is ``max (src, 0)``.

.. code-block:: c++

    vFloat approx_recip (vFloat src, RecipMode = RecipMode::All);
    vFloat approx_exp (vFloat src);
    vFloat approx_sqrt (vFloat src);
    vFloat approx_tanh (vFloat src);

Compute approximate reciprocals, exponentials, square roots and
hyperbolic tangents. ``RecipMode`` may be ``All`` or ``IfNegative``.
Not available on Wormhole, and `sqrt` and `tanh` not available on
Blackhole.

.. code-block:: c++

    vFloat lut(const vFloat v, const vUInt l0, const vUInt l1, const vUInt l2, const int offset)
    vFloat lut_sign(const vFloat v, const vUInt l0, const vUInt l1, const vUInt l2, const int offset)

``l0``, ``l1``, ``l2`` each contain 2 8-bit floating point values ``A`` and ``B`` with ``A`` in bits 15:8 and ``B`` in bits 7:0. The 8-bit format is:

  * 0xFF represents the value 0, otherwise
  * bit[7] is the sign bit, bit[6:4] is the unsigned exponent_extender and bit[3:0] is the mantissa

Floating point representations of ``A`` and ``B`` (19-bit on GS and 32-bit on WH) are constructed by:

  * Using the sign bit
  * Generating an 8-bit exponent as (127 – exponent_extender)
  * Generating a mantissa by padding the right of the specified 4 bit mantissa with 0s

``A`` and ``B`` are selected from one of ``l0``, ``l1`` or ``l2`` based on the value in ``v`` as follows:

  * ``l0`` when ``v`` < 0
  * ``l1`` when ``v`` == 0
  * ``l2`` when ``v`` > 0

.. XXXX is this backwards?
.. Returns the result of the computation ''A * ABS(v) + B''.  The ''lut_sgn'' variation discards the calculated sign bit and instead uses the sign of ''v''.

.. code-block:: c++

    Vec subvec_shflror1(Vec& v)
    Vec subvec_shflshr1(Vec& v)

.. code-block:: c++

    void subvec_transp(Vec& A, Vec& B, Vec& C, Vec& D)

Assigning and Using Constant Registers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Programmable constant registers are accessed and assigned just
like any other variables, for example:

.. code-block:: c++

    vConstFloatPrgm0 = 3.14159265;
    vFloat two_pi = 2.0f * vConstFloatPrgm0;

Writing to a constant register first loads the constant into a temporary LReg
then assigns the LReg to the constant register and so takes 1 cycle longer
than just loading an LReg.  Accessing a constant register is just as fast as
accessing an LReg.  Loading a constant register loads the same value into all
vector elements.

Using programmable constants reduces the mount of loads needed during kernel
execution and so can improve performance. However, users should be aware that
other functions may overwrite the constant registers (this is what some of the
``init_*`` functions do).  Therefore, if a constant register is used, it
should be placed in the initialization function and users needs to ensure
no other function overwrites it before use.

Assigning LRegs
^^^^^^^^^^^^^^^

Some highly optimized code may call a function prior to the kernel to
pre-load values into specific LRegs and then access those values in
the kernel.  Note that if the register's value must be preserved when
the kernel exits, you must restore the value explicitly by assigning
back into the LReg.

For example:

.. code-block:: c++

    vFloat x = l_reg[LRegs::LReg1];  // x is now LReg1
    vFloat y = x + 2.0f;
    l_reg[LRegs::LReg1] = x;         // this is necessary at the end of the function
                                     // to preserve the value in LReg1 (if desired)

You may mark an lreg as used in code that the compiler cannot examine
with the ``used`` function:

.. code-block:: c++

    l_reg[LRegs::LReg0].used();
    // your code here

The compiler will not keep a value live in  lreg0 across your code.

Miscellaneous
=============

Register Pressure Management
----------------------------

Note that the wrapper introduces temporaries in a number of places.  For
example:

.. code-block:: c++

  dst_reg[0] = vFloat(dst_reg[0]) + vFloat(dst_reg[1]);

loads dst_reg[0] and dst_reg[1] into temporary LREGs (as expected).

The compiler cannot spill registers (there is no hardware mechanism to
do so).  Exceeding the number of registers available will result in
the cryptic: ``error: cannot store SFPU register (register spill?) -
exiting!`` without a line number.

The compiler does a reasonable job with lifetime analysis when assigning
variables to registers.  Reloading or recalculating results helps the compiler
free up and reuse registers and is a good way to correct a spilling error.

Optimizer
---------

There is a basic optimizer in place.  The optimization philosophy to date is to enable the programmer
to write optimal code.  This is different from mainstream compilers which may generate optimal code
given non-optimal source.  For example, common sub-expression elimination and the like are not
implemented.  The optimizer will handle the following items:

  * MAD generation (from MUL/ADD)
  * MULI, ADDI generation (from MUL + const, or ADD + const)
  * Swapping the order of arguments to instructions that use the destination-as-source, e.g., SFPOR to minimize the need for register moves
  * CC enables (PUSHC, POPC, etc.)
  * Instruction combining for comparison operations.  For example, a subtract of 5 followed by a compare against 0 gets combined into one operation
  * NOP insertion for instructions which must be followed by an independent instruction or ``SFPNOP``. Note that this pass (presently) does not move instructions to fill the slot but will skip adding a ``SFPNOP`` if the next instruction is independent. In other words, reordering your code to reduce dependent chains of instructions may improve performance

There is a potential pitfall in the above in that the MAD generator could
change code which would not run out of registers with, say, a MULI followed by
an ADDI into code that runs out of registers with a MAD.  (future todo to fix this).

SFPREPLAY
---------

The ``SFPREPLAY`` instruction available on Wormhole and Blackhole allows the RISCV processor
to submit up to 32 SFP instructions at once.  The compiler looks for sequences
of instructions that repeat, stores these and then "replays" them later.

The current implementation of this is very much first cut: it does not handle
kernels with rolled up loops very well.  Best performance is typically attained by
unrolling the top level loop and then letting the compiler find the repetitions
and replace them with ``SFPREPLAY``.  This works well when the main loop
contains < 32 instructions, but performance starts to degrade again as the
number of instructions grows.

The other issue that can arise with ``SFPREPLAY`` is that sometimes the last
unrolled loop of instructions uses different registers than the prior
loops resulting in imperfect utilization of the replay.


Tools
-----

The `sfpi  repository<https://github.com/tenstorrent/sfpi>` contains a ``tools`` directory.  ``cd`` into that directory and
type ``make`` to build ``fp16c`` which is a converter that converts floating point
values to fp16a, fp16b and the LUT instruction's fp8 as well as the other way
(integer to float/fp16a/fp16b/fp8).  This is useful for writing optimal code or
looking through assembly dumps.

Pitfalls/Oddities/Limitations
=============================

Arrays/Storing to Memory
------------------------
The SFPU can only read/write vectors to/from the destination register, it
cannot read/write them to memory.  Therefore, SFPI does not support arrays of
vectors.  Using arrays may work if the optimizer is able to optimize out the
loads/stores, however, this is brittle and so is not recommended.  Storing a
vector to memory will result in an error similar to the following:

.. code-block:: c++

    tt-metal/tt_metal/hw/ckernels/sfpi/include/sfpi.h:792:7: error: cannot write sfpu vector to memory
      792 |     v = (initialized) ? __builtin_rvtt_sfpassign_lv(v, in) : in;
          |       ^
    /tt-metal/tt_metal/hw/ckernels/sfpi/include/sfpi.h:792:7: error: cannot write sfpu vector to memory


Function Calls
--------------

There is no ABI and none of the vector types can be passed on the stack.
Therefore, all function calls must be inlined.  To ensure this use
``sfpi_inline``, which is defined to ``__attribute__((always_inline))`` on GCC.

Register Spilling
-----------------

The compiler does not implement register spilling.  Since there are only 8 general purpose
LRegs, running out of registers is not an uncommon occurrence.  If you see the
following: ``error: cannot store SFPU register (register spill?) - exiting!``
you have most likely run out of registers.

You can potentially spill registers by storing values to ``l_reg[]`` and
reloading them later. However this is not done automatically via the compiler
as it does not know which of ``l_reg[]`` values need to be preserved.

Error Messages
--------------

Unfortunately, many errors are attributed to the code in the wrapper rather than in the code
being written.  For example, using an uninitialized variable would show an error at a macro
called by a wrapper function before showing the line number in the user's code.

Limitations
-----------

  * Forgetting a ``v_endif`` results in mismatched {} error which can be confusing (however, catches the case where a ``v_endif`` is missing!)
  * In general, incorrect use of vector operations (e.g., accidentally using a scalar argument instead of a vector) results in warnings/errors within the wrapper rather than in the calling code
  * Keeping too many variables alive at once requires register spilling which is not implemented and causes a compiler abort
  * The gcc compiler occasionally moves a value from one register to another for no apparent reason.  At this point it appears there is nothing that can be done about this besides hoping that the issue is fixed in a future version of gcc.
