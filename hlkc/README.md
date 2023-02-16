# HLK Compiler (HLKC)

Translates HLK kernels into LLK-API-based Ckernels.

## Rose Installation (a tool used by HLKC)

HLKC has a dependency on the ROSE compiler/tools and it requires its installation. Ubuntu 20.04 instructions are below.

```shell
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:rosecompiler/rose-development
sudo apt-get install rose
sudo apt-get install rose-tools
```

`rose-stable` was not available for Ubuntu 20.04 in early April 2021. It will probably appear soon, for now `rose-development` is getting the job done.

Instructions reproduced from here: https://github.com/rose-compiler/rose/wiki/Install-Using-apt-get

## HLKC Build
Go to `ll-sw/hlkc`.

```shell
make hlkc
```
Alternatively, use `make hlkc_fast` (it's just a faster `hlkc` build, however paths are hard-coded so it's not as portable).


## HLKC Usage
See below examples of how to run it. Currently 4 seperate calls are required to generate upack, math, pack LLKs and the struct-init-generator.

Run from the `ll-sw/hlkc` folder:

Matmul:
```
./hlkc ../hlks/matmul.cpp -hlkc:llk_target unpack -rose:o out/llks/matmul_unpack.cpp
./hlkc ../hlks/matmul.cpp -hlkc:llk_target math -rose:o out/llks/matmul_math.cpp
./hlkc ../hlks/matmul.cpp -hlkc:llk_target pack -rose:o out/llks/matmul_pack.cpp
./hlkc ../hlks/matmul.cpp -hlkc:llk_target struct_init_gen -rose:o out/llks/matmul_struct_init_gen.cpp
```

Unary:
```
./hlkc ../hlks/eltwise_unary_datacopy_full_dst_mode.cpp -hlkc:llk_target unpack -rose:o out/llks/eltwise_unary_unpack.cpp
./hlkc ../hlks/eltwise_unary_datacopy_full_dst_mode.cpp -hlkc:llk_target math -rose:o out/llks/eltwise_unary_math.cpp
./hlkc ../hlks/eltwise_unary_datacopy_full_dst_mode.cpp -hlkc:llk_target pack -rose:o out/llks/eltwise_unary_pack.cpp
./hlkc ../hlks/eltwise_unary_datacopy_full_dst_mode.cpp -hlkc:llk_target struct_init_gen -rose:o out/llks/eltwise_unary_struct_init_gen.cpp
```

## HLKC Cache
The result of each hlk compile is cached in a directory under `./.hlkc_cache`.
The current caching algorithm is simple, a 128-bit hash value will be created from the `hlkc` executable and hlk kernel file contents.
This means that as long as the `hlkc` executable and hlk kernel file does not change, a recompilation will be simply retrieving the cached files from the cache dir.
The cache only stores the latest version of the compilation, any older versions will be deleted.

For compilation command `./hlkc ../hlks/matmul.cpp -hlkc:llk_target unpack -rose:o out/llks/matmul_unpack.cpp`, the cache directory will be `./.hlkc_cache/matmul.cpp/unpack/<config_flags>/<hash_value>/chlkc_unpack.cpp/`

When the compilation flag is changed, i.e. perf\_dump\_en or untilize\_output, a new cache directory is created even if the `hlkc` executable and hlk kernel file remains the same.

### Disabling Cache

To disable caching on compilation at the `hlkc` level, add the flag `-cache:off`.

To disable caching on compilation at the test level, add the flag `--disable-cache`.

When cache is disabled, compilation will ignore any entries in the cache and it will also not update the cache with new entries.
