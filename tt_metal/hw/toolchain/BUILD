package(default_visibility = ["//visibility:public"])
exports_files(["tensix.ld", "tmu-crt0.S", "substitutes.cpp", "ncrisc.ld"])



filegroup(
    name = "ld-files",
    srcs = ["tensix-sections.ld", "tensix-address.ld", "tensix-memory.ld", "idle-erisc-memory.ld"]
)

py_binary(
    name = "hex8tohex32",
    srcs = ["hex8tohex32.py"],
)


py_binary(
    name = "fwlog",
    srcs = ["fwlog.py"],
)
