from collections.abc import Callable
from dataclasses import dataclass, field

from helpers.format_config import FormatConfig, List
from helpers.llk_params import MathOperation
from helpers.test_variant_parameters import TemplateParameter

DispatchDefinesFn = Callable[[FormatConfig], dict]


@dataclass
class BaseOpSpec:
    """Declarative description of one SFPU op for the registry-driven test.

    HOW TO ADD AN OP
    ----------------
    Drop a module under ``quasar/sfpu/{unary,binary,ternary}/`` that defines a
    module-level ``SPEC = <Unary|Binary|Ternary>OpSpec(...)``. It is auto-discovered
    (see ``quasar/sfpu/__init__.py``) and ``quasar/test_sfpu.py`` sweeps it across the
    valid (format, dest_acc) combinations and runs it through the shared C++ skeleton
    ``sources/quasar/sfpu_test.cpp``. For the common case (a plain elementwise op with
    a standard golden) you write *only* the spec — no new test or kernel code.

    The skeleton gathers operands from DEST tiles 0/1/2 and writes the result to tile
    0; you describe the op via the fields below.

    FIELDS
    ------
    name : str
        Unique short name; shows up in the pytest variant id.

    math_op : MathOperation
        The op's MathOperation. Selects the default golden — UnarySFPUGolden /
        BinarySFPUGolden / TernarySFPUGolden keyed on this op — unless ``golden`` is set.

    include_header : str
        The op's C++ header, as written in an ``#include``, relative to either the LLK
        common tree or tt-metal's quasar llk_api (both are on the compiler's -I path):
          - "sfpu/ckernel_sfpu_square.h"        (LLK common/inc)
          - "llk_sfpu/ckernel_sfpu_binary.h"    (metal llk_api, the sfpi float ops)
        Discovery verifies the file exists under one of those roots.

    sfpu_defines : (FormatConfig) -> dict
        Returns the per-op ``#define``s emitted into the build header. Always set:
          SFPU_OP_CALL         - the ckernel::sfpu callable to dispatch, e.g.
                                 "ckernel::sfpu::_calculate_square_" (template args and
                                 build-time constants like is_fp32_dest_acc_en are fine).
        Optional keys:
          SFPU_INIT            - a statement run once before the op call, e.g.
                                 "ckernel::sfpu::_init_gelu_();" (ops needing setup).
          SFPU_ADDITIONAL_ARGS - extra trailing args appended to the op call, WITH a
                                 leading comma, e.g. ", 2.0f".
          SFPU_TYPE            - SfpuType for ops whose init is templated on it
                                 (ternary init only, e.g. "ckernel::SfpuType::where").
          SFPU_BINARY_OPERANDS - override the default binary operand indices "0u, 1u, 0u"
                                 for ops that index by element OFFSET rather than tile
                                 index, e.g. "0, tile_stride, 0" (the int ops). ``tile_stride``
                                 is a kernel-local for these to reference.
        It takes the FormatConfig so the defines can vary by format if ever needed.

    extra_templates : list[TemplateParameter]
        Extra build-time template params to emit (e.g. ternary's VECTOR_MODE face
        selector, or a format-baked constexpr the op needs). Spliced into the variant's
        templates. Give a TemplateParameter a list-valued field to SWEEP it: the driver
        expands the cross product into separate variants (one pytest id each), e.g.
        ``VECTOR_MODE([VectorMode.RC, VectorMode.R])`` runs two variants. Sweeping is
        wired for the ternary arity (the only one whose dispatch consumes such a param
        today); a swept axis that changes the result needs a matching ``golden`` hook.

    stimuli : StimuliSpec | None
        Overrides the driver's default per-dtype input range. Use when the op needs a
        specific domain (e.g. sqrt: positive only; divide: bounded away from zero).

    prepare : Callable | None
        Post-processes the generated stimulus tensor BEFORE it is used (for both the
        golden and the device), returning the modified tensor. Use to inject operand
        lanes a random range can't hit (e.g. divide's 0/0 -> NaN, x/0 -> +/-Inf).
        Signature (keyword-only):
            prepare(src, *, src_tiles) -> Tensor
        where ``src_tiles`` is the tuple of DEST tile indices the operands occupy for
        this variant — (0,) for unary, (src0, src1) for binary (these vary with the
        tile-layout sweep), (0, 1, 2) for ternary. A tile is 1024 elements; index into
        ``src.flatten()`` as ``tile * 1024 + lane``.

    golden : Callable | None
        Overrides the default math_op-keyed golden. Use for ops whose reference isn't
        ``<Arity>SFPUGolden(math_op)`` — fused ops (e.g. swiglu) or ops needing a scalar
        the standard generator doesn't take. Signature (keyword-only):
            golden(*, src, io, dest_acc, dimensions) -> Tensor
        where ``src`` is the prepared stimulus tensor, ``io`` the InputOutputFormat,
        ``dest_acc`` the DestAccumulation, ``dimensions`` the [rows, cols] input dims.
        Return the flat, output-format golden for the packed result (one result tile for
        binary/ternary; the transformed tiles for unary).

    verify : Callable | None
        Optional extra assertion run on the packed result AFTER the tolerance-based
        golden comparison. Use for op-specific exactness checks tolerance can't express
        (e.g. divide forces x/x to a bit-exact 1.0). Signature (keyword-only):
            verify(res_tensor, *, io) -> None
        Raise AssertionError on failure; the return value is ignored.
    """

    name: str
    math_op: MathOperation
    include_header: str
    sfpu_defines: DispatchDefinesFn
    extra_templates: List[TemplateParameter] = field(default_factory=list)
    stimuli: object = None
    prepare: object = None
    golden: object = None
    verify: object = None

    def __post_init__(self):
        if type(self) is BaseOpSpec:
            raise TypeError(
                "BaseOpSpec is not a concrete implementation; use a subclass"
            )
