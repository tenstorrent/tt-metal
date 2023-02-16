# Low Level Buda

A low level programming model with user facing [host](./api/host_api.hpp) APIs 

## Running tests

```
$ make clean
$ make build
$ make device

$ export BUDA_HOME=<this repo dir>

$ make ll_buda/tests
$ ./build/test/ll_buda/tests/test_add_two_ints
```

