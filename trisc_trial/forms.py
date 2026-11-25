"""build.h statement forms. Each form: render(n, v) -> (prelude, build_h, uses).

n: fresh identifiers (n.A, n.B, ...), v: list of small ints (variant values). `uses` are C++ expressions
(int-valued) a TRISC could use; the harness picks a random subset. `prelude` goes in a header included
before build.h (stands in for LLK headers), so it is outside build.h.
"""

F = {}


def form(nvals, cls):
    def deco(fn):
        F[fn.__name__] = (nvals, cls, fn)
        return fn

    return deco


@form(1, "real")
def simple_u32(n, v):
    return "", f"constexpr std::uint32_t {n.A} = {v[0]}u;", [n.A]


@form(1, "real")
def simple_bool(n, v):
    return "", f"constexpr bool {n.A} = {'true' if v[0] % 2 else 'false'};", [f"(int){n.A}"]


@form(1, "real")
def enum_value(n, v):
    return (f"enum class {n.E} {{ X0, X1, X2, X3 }};",
            f"constexpr {n.E} {n.A} = {n.E}::X{v[0] % 4};", [f"(int){n.A}"])


@form(1, "real")
def scoped_value(n, v):
    return (f"namespace {n.N} {{ constexpr int K0 = 0, K1 = 1, K2 = 2, K3 = 3; }}",
            f"constexpr std::uint32_t {n.A} = {n.N}::K{v[0] % 4};", [n.A])


@form(2, "real")
def multiline_array(n, v):
    return ("", f"constexpr Arr<int, 2> {n.A} = {{\n    {v[0]},\n    {v[1]}\n}};", [f"{n.A}.d[1]"])


@form(2, "real")
def struct_then_array(n, v):
    return ("", f"// {n.A} config\nstruct {n.S}\n{{\n    int a;\n    int b;\n    constexpr {n.S}(int a_, int b_) : a(a_), b(b_) {{}}\n}};\n"
            f"constexpr Arr<{n.S}, 1> {n.A} = {{{n.S}({v[0]}, {v[1]})}};", [f"{n.A}.d[0].b"])


@form(1, "real")
def comment_before(n, v):
    return "", f"// {n.A}\nconstexpr int {n.A} = {v[0]};", [n.A]


@form(1, "real")
def trailing_space(n, v):
    return "", f"constexpr std::uint8_t {n.A} = {v[0]}u; ", [n.A]


@form(1, "real")
def runtime_struct(n, v):
    return "", f"struct {n.S}\n{{\n    int f{v[0]};\n}};", [f"(int)sizeof({n.S})"]


@form(1, "real")
def define_flag(n, v):
    return "", (f"#define {n.M} {v[0]}" if v[0] % 2 else ""), [f"{n.M}_get()"] if False else ["0"]


@form(1, "real")
def define_used(n, v):
    return "", f"#define {n.M} {v[0]}", [n.M]


@form(2, "real")
def closure(n, v):
    return "", f"constexpr int {n.A} = {v[0]};\nconstexpr int {n.B} = {n.A} + {v[1]};", [n.B]


@form(2, "decl")
def multi_declarator(n, v):
    return "", f"constexpr int {n.A} = {v[0]}, {n.B} = {v[1]};", [n.B]


@form(2, "decl")
def two_on_a_line(n, v):
    return "", f"constexpr int {n.A} = {v[0]}; constexpr int {n.B} = {v[1]};", [n.B]


@form(2, "decl")
def trailing_comment(n, v):
    return "", f"constexpr int {n.A} = {v[0]}; // note\nconstexpr int {n.B} = {v[1]};", [n.B]


@form(1, "decl")
def struct_declarator(n, v):
    return "", f"constexpr struct {n.S} {{ int v; }} {n.A}{{{v[0]}}};", [f"{n.A}.v"]


@form(1, "decl")
def function_noexcept(n, v):
    return "", f"constexpr int {n.A}() noexcept {{ return {v[0]}; }}", [f"{n.A}()"]


@form(1, "decl")
def function_default_arg(n, v):
    return "", f"constexpr int {n.A}(int {n.P} = {v[0]}) {{ return {n.P}; }}", [f"{n.A}()"]


@form(1, "decl")
def function_trailing(n, v):
    return "", f"constexpr auto {n.A}() -> long {{ return {v[0]}; }}", [f"(int){n.A}()"]


@form(1, "decl")
def operator_eq(n, v):
    return (f"struct {n.S} {{ int v; }};",
            f"constexpr bool operator==({n.S} a, {n.S} b) {{ return a.v + {v[0]} == b.v; }}",
            [f"(int)({n.S}{{1}} == {n.S}{{2}})"])


@form(1, "decl")
def udl(n, v):
    return "", f"constexpr int operator\"\"_{n.A}(unsigned long long x) {{ return (int)x + {v[0]}; }}", [f"3_{n.A}"]


@form(1, "decl")
def brace_in_type(n, v):
    return (f"struct {n.T} {{ constexpr operator int() const {{ return 1; }} }};\ntemplate <int K> struct {n.S} {{ int v; }};",
            f"constexpr {n.S}<{n.T}{{}}> {n.A} = {{{v[0]}}};", [f"{n.A}.v"])


@form(1, "safe")
def brace_init(n, v):
    return "", f"constexpr int {n.A}{{{v[0]}}};", [n.A]


@form(1, "safe")
def static_inline(n, v):
    return "", f"static constexpr int {n.A} = {v[0]};\ninline constexpr int {n.B} = {v[0]} + 1;", [n.A, n.B]


@form(1, "safe")
def paren_init(n, v):
    return "", f"constexpr int {n.A}({v[0]});", [n.A]


@form(1, "safe")
def c_array(n, v):
    return "", f"constexpr int {n.A}[] = {{{v[0]}, 2}};", [f"{n.A}[0]"]


@form(1, "safe")
def attribute(n, v):
    return "", f"[[maybe_unused]] constexpr int {n.A} = {v[0]};", [n.A]


@form(1, "safe")
def const_var(n, v):
    return "", f"const int {n.A} = {v[0]};\ninline const int {n.B} = {v[0]};", [n.A, n.B]


@form(1, "safe")
def enum_decl(n, v):
    return "", f"enum class {n.E} {{ X = {v[0]} }};", [f"(int){n.E}::X"]


@form(1, "safe")
def namespace_block(n, v):
    return "", f"namespace {n.N} {{\nconstexpr int {n.A} = {v[0]};\n}}", [f"{n.N}::{n.A}"]


@form(1, "safe")
def using_alias(n, v):
    return "", f"using {n.T} = Arr<int, {v[0] + 1}>;", [f"(int)sizeof({n.T})"]


@form(1, "safe")
def var_template(n, v):
    return "", f"template <int N> constexpr int {n.A} = N * {v[0]};", [f"{n.A}<2>"]


@form(1, "safe")
def specialisation(n, v):
    return (f"template <int N> struct {n.S} {{ static constexpr int v = 0; }};",
            f"template <> struct {n.S}<1> {{ static constexpr int v = {v[0]}; }};", [f"{n.S}<1>::v"])


@form(1, "safe")
def string_value(n, v):
    return "", f"constexpr const char* {n.A} = \"s{v[0]},;{{\";", [f"{n.A}[1]"]


@form(1, "safe")
def template_comma(n, v):
    return (f"template <int X, int Y> struct {n.S} {{ static constexpr int value = X + Y; }};",
            f"constexpr int {n.A} = {n.S}<{v[0]}, 2>::value;", [n.A])


@form(2, "safe")
def ifdef_choice(n, v):
    return "", (f"{'#define ' + n.M if v[1] % 2 else ''}\n#ifdef {n.M}\nconstexpr int {n.A} = {v[0]};\n"
                f"#else\nconstexpr int {n.A} = {v[0]} + 7;\n#endif"), [n.A]


@form(1, "safe")
def sizeof_type(n, v):
    return "", f"struct {n.S} {{ char c[{v[0] + 1}]; }};\nconstexpr int {n.A} = 0;", [f"(int)sizeof({n.S})"]


@form(2, "macro")
def macro_outside_to_build_name(n, v):
    return (f"#define {n.M} {n.B}",
            f"constexpr int {n.B} = {v[0]};\nconstexpr int {n.A} = {n.M} + {v[1]};", [n.A])


@form(1, "macro")
def macro_is_declared_name(n, v):
    return f"#define {n.M} {n.A}", f"constexpr int {n.M} = {v[0]};", [n.A]


@form(2, "macro")
def macro_paste(n, v):
    return ("#define TT_CAT(a, b) a##b",
            f"constexpr int {n.A}x{n.B} = {v[0]};\nconstexpr int {n.C} = TT_CAT({n.A}x, {n.B}) + {v[1]};", [n.C])


@form(2, "macro")
def macro_body_declares(n, v):
    return (f"#define {n.M}(x) x, {n.B} = x + 1",
            f"constexpr int {n.A} = {n.M}({v[0]});", [n.B])


@form(2, "macro")
def macro_in_build_used_by_unnamed(n, v):
    return (f"#define {n.M} {n.B}",
            f"constexpr int {n.B} = {v[0]};\nstruct {n.S} {{ int a[{n.M} + {v[1]} + 1]; }};", [f"(int)sizeof({n.S})"])


@form(2, "macro")
def define_in_build_to_name(n, v):
    return "", f"constexpr int {n.A} = {v[0]};\n#define {n.M} ({n.A} + {v[1]})", [n.M]


@form(2, "directive")
def indented_define(n, v):
    return "", f"constexpr int {n.A} = {v[0]}; // c\n  #define {n.M} {v[1]}\nconstexpr int {n.B} = 0;", [n.M]


@form(2, "directive")
def define_continuation(n, v):
    return "", f"#define {n.M} \\\n    {v[0]}\nconstexpr int {n.A} = {v[1]};", [n.M]


@form(1, "codegen")
def address_of_template(n, v):
    return (f"template <int N> int {n.T}() {{ return N * 3; }}",
            f"constexpr auto {n.A} = &{n.T}<{v[0]}>;", ["0"])


@form(1, "codegen")
def comment_block(n, v):
    return "", f"/*\nconstexpr int {n.A} = {v[0]};\n*/\nconstexpr int {n.B} = 1;", [n.B]


@form(1, "diag")
def ill_formed_unused(n, v):
    return "", f"constexpr int {n.A} = 10 / {v[0] % 2};", ["0"]


@form(1, "diag")
def static_assert_merged(n, v):
    return "", f"constexpr int {n.A} = 1; static_assert({v[0] % 2} == 1);", ["0"]


@form(1, "position")
def directive_order(n, v):
    if v[0] % 2:
        return "", f"#define {n.M} 1\nconstexpr int {n.A} = {n.M};\n#undef {n.M}\n#define {n.M} 2", [n.A]
    return "", f"#define {n.M} 1\n#undef {n.M}\n#define {n.M} 2\nconstexpr int {n.A} = {n.M};", [n.A]


@form(1, "position")
def line_after_dropped(n, v):
    return "", f"constexpr Arr<int, 2> {n.B} = {{\n{chr(10) * (v[0] % 2)}1, 2}};\nconstexpr int {n.A} = __LINE__;", [n.A]


@form(1, "position")
def counter_in_dropped(n, v):
    return "", f"constexpr int {n.B} = {'__COUNTER__' if v[0] % 2 else '0'};\nconstexpr int {n.A} = __COUNTER__;", [n.A]


@form(1, "codegen")
def virtual_type(n, v):
    return (f"template <int K> struct {n.T} {{ constexpr {n.T}() {{}} virtual int f() const {{ return K * 5; }} }};",
            f"constexpr {n.T}<{v[0]}> {n.A}{{}};", ["0"])


@form(2, "split")
def raw_string(n, v):
    return "", f'constexpr const char* {n.A} = R"(\nx;\nconstexpr int {n.B} = {v[0]};\ny)";', [f"[]{{ int s = {v[1]}; for (const char* p = {n.A}; *p; ++p) s += *p; return s; }}()"]


@form(2, "directive")
def digraph_directive(n, v):
    return "", f"constexpr int {n.A} =\n%:define {n.M} {v[0]}\n{v[1]};", [n.M]

