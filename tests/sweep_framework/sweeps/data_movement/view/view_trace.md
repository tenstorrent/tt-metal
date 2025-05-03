### aten.view.default
|      | ATen Input Variations                                                          | Status   | Single-native-run   | Single-run   | Single-accuracy   | Single-converted   |
|-----:|:-------------------------------------------------------------------------------|:---------|:--------------------|:-------------|:------------------|:-------------------|
|    0 | Tensor<[0, 1, 4]> self = ?,<br>List[int] size = [0, 4]                         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|    1 | Tensor<[0, 2, 2]> self = ?,<br>List[int] size = [0, 4]                         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|    2 | Tensor<[1, 1, 1, 16, 2]> self = ?,<br>List[int] size = [1, 1, 1, 32]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|    3 | Tensor<[1, 1, 1, 4, 4]> self = ?,<br>List[int] size = [1, -1, 4]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|    4 | Tensor<[1, 1, 1, 4, 91]> self = ?,<br>List[int] size = [1, -1, 91]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|    5 | Tensor<[1, 1, 1, 6, 4]> self = ?,<br>List[int] size = [1, -1, 4]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|    6 | Tensor<[1, 1, 1, 6, 91]> self = ?,<br>List[int] size = [1, -1, 91]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|    7 | Tensor<[1, 1, 1, 7, 7, 1024]> self = ?,<br>List[int] size = [1, 49, 1024]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|    8 | Tensor<[1, 1, 1, 7, 7, 768]> self = ?,<br>List[int] size = [1, 49, 768]        | Done     | N/A                 | N/A          | N/A               | N/A                |
|    9 | Tensor<[1, 1, 1, 8, 8, 1024]> self = ?,<br>List[int] size = [1, 64, 1024]      | None     | N/A                 | N/A          | N/A               | N/A                |
|   10 | Tensor<[1, 1, 1, 8, 8, 768]> self = ?,<br>List[int] size = [1, 64, 768]        | None     | N/A                 | N/A          | N/A               | N/A                |
|   11 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [-1, 1024]                  | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   12 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   13 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, 1, 1024]                | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   14 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, 1, 16, 64]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   15 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, 1024]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   16 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1024]                      | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   17 | Tensor<[1, 1, 12, 16, 2]> self = ?,<br>List[int] size = [1, 192, 2]            | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   18 | Tensor<[1, 1, 12, 16]> self = ?,<br>List[int] size = [1, 192]                  | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   19 | Tensor<[1, 1, 12, 64]> self = ?,<br>List[int] size = [1, -1, 768]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   20 | Tensor<[1, 1, 12, 64]> self = ?,<br>List[int] size = [1, 1, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|   21 | Tensor<[1, 1, 1280]> self = ?,<br>List[int] size = [1, 1280]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   22 | Tensor<[1, 1, 16, 16, 2]> self = ?,<br>List[int] size = [1, 1, 16, 32]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   23 | Tensor<[1, 1, 16, 64]> self = ?,<br>List[int] size = [1, -1, 1024]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   24 | Tensor<[1, 1, 16, 64]> self = ?,<br>List[int] size = [1, 1, 1024]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   25 | Tensor<[1, 1, 16384, 256]> self = ?,<br>List[int] size = [1, 16384, 256]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|   26 | Tensor<[1, 1, 16384, 32]> self = ?,<br>List[int] size = [1, 16384, 32]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|   27 | Tensor<[1, 1, 19200, 300]> self = ?,<br>List[int] size = [1, 19200, 300]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|   28 | Tensor<[1, 1, 19200, 64]> self = ?,<br>List[int] size = [1, 19200, 64]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|   29 | Tensor<[1, 1, 2048]> self = ?,<br>List[int] size = [1, 2048]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   30 | Tensor<[1, 1, 256, 32]> self = ?,<br>List[int] size = [1, 256, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   31 | Tensor<[1, 1, 256]> self = ?,<br>List[int] size = [1, 256]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   32 | Tensor<[1, 1, 256]> self = ?,<br>List[int] size = [256]                        | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   33 | Tensor<[1, 1, 300, 64]> self = ?,<br>List[int] size = [1, 300, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   34 | Tensor<[1, 1, 3072]> self = ?,<br>List[int] size = [1, 1, 4, -1]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   35 | Tensor<[1, 1, 3072]> self = ?,<br>List[int] size = [1, 3072]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|   36 | Tensor<[1, 1, 32, 256]> self = ?,<br>List[int] size = [1, 32, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   37 | Tensor<[1, 1, 384]> self = ?,<br>List[int] size = [1, -1, 6, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   38 | Tensor<[1, 1, 384]> self = ?,<br>List[int] size = [1, 384]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   39 | Tensor<[1, 1, 384]> self = ?,<br>List[int] size = [384]                        | Done     | N/A                 | N/A          | N/A               | N/A                |
|   40 | Tensor<[1, 1, 4, 256]> self = ?,<br>List[int] size = [1, 1, 4, 4, 64]          | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   41 | Tensor<[1, 1, 4096]> self = ?,<br>List[int] size = [1, 4096]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   42 | Tensor<[1, 1, 45]> self = ?,<br>List[int] size = [1, 45]                       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   43 | Tensor<[1, 1, 512]> self = ?,<br>List[int] size = [1, -1, 8, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   44 | Tensor<[1, 1, 512]> self = ?,<br>List[int] size = [1, 512]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   45 | Tensor<[1, 1, 6, 64]> self = ?,<br>List[int] size = [1, -1, 384]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   46 | Tensor<[1, 1, 64, 300]> self = ?,<br>List[int] size = [1, 64, 300]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   47 | Tensor<[1, 1, 7, 1, 7, 1024]> self = ?,<br>List[int] size = [1, 7, 7, 1024]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|   48 | Tensor<[1, 1, 7, 1, 7, 768]> self = ?,<br>List[int] size = [1, 7, 7, 768]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|   49 | Tensor<[1, 1, 7, 64]> self = ?,<br>List[int] size = [1, 1, 7, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|   50 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [-1, 1, 768]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   51 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|   52 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [1, 1, 12, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|   53 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [1, 768]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|   54 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [768]                        | Done     | N/A                 | N/A          | N/A               | N/A                |
|   55 | Tensor<[1, 1, 8, 1, 8, 1024]> self = ?,<br>List[int] size = [1, 8, 8, 1024]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|   56 | Tensor<[1, 1, 8, 1, 8, 768]> self = ?,<br>List[int] size = [1, 8, 8, 768]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|   57 | Tensor<[1, 1, 8, 64]> self = ?,<br>List[int] size = [1, -1, 512]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   58 | Tensor<[1, 1, 80]> self = ?,<br>List[int] size = [1, 80]                       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   59 | Tensor<[1, 10, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|   60 | Tensor<[1, 10, 1024]> self = ?,<br>List[int] size = [10, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|   61 | Tensor<[1, 10, 12, 64]> self = ?,<br>List[int] size = [1, -1, 768]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   62 | Tensor<[1, 10, 12, 64]> self = ?,<br>List[int] size = [1, 10, 768]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   63 | Tensor<[1, 10, 16, 64]> self = ?,<br>List[int] size = [1, -1, 1024]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|   64 | Tensor<[1, 10, 2048]> self = ?,<br>List[int] size = [10, 2048]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|   65 | Tensor<[1, 10, 3072]> self = ?,<br>List[int] size = [10, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|   66 | Tensor<[1, 10, 4096]> self = ?,<br>List[int] size = [10, 4096]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|   67 | Tensor<[1, 10, 512]> self = ?,<br>List[int] size = [1, -1, 8, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|   68 | Tensor<[1, 10, 512]> self = ?,<br>List[int] size = [10, 512]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|   69 | Tensor<[1, 10, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   70 | Tensor<[1, 10, 768]> self = ?,<br>List[int] size = [1, 10, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   71 | Tensor<[1, 10, 768]> self = ?,<br>List[int] size = [10, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|   72 | Tensor<[1, 10, 8, 64]> self = ?,<br>List[int] size = [1, -1, 512]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|   73 | Tensor<[1, 100, 192]> self = ?,<br>List[int] size = [100, 192]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|   74 | Tensor<[1, 1000, 1, 1]> self = ?,<br>List[int] size = [1, 1000]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|   75 | Tensor<[1, 1000]> self = ?,<br>List[int] size = [1, 1000, 1, 1]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|   76 | Tensor<[1, 1000]> self = ?,<br>List[int] size = [1000]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|   77 | Tensor<[1, 1008, 1, 1]> self = ?,<br>List[int] size = [1, 1008]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|   78 | Tensor<[1, 1024, 1, 1]> self = ?,<br>List[int] size = [1, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|   79 | Tensor<[1, 1024, 1, 1]> self = ?,<br>List[int] size = [1, 1024]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|   80 | Tensor<[1, 1024, 14, 14]> self = ?,<br>List[int] size = [1, 1024, 196]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|   81 | Tensor<[1, 1024, 16, 16]> self = ?,<br>List[int] size = [1, 1024, 256]         | None     | N/A                 | N/A          | N/A               | N/A                |
|   82 | Tensor<[1, 1024, 160]> self = ?,<br>List[int] size = [1, 1024, 5, 32]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|   83 | Tensor<[1, 1024, 160]> self = ?,<br>List[int] size = [1, 32, 32, -1]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|   84 | Tensor<[1, 1024, 160]> self = ?,<br>List[int] size = [1024, 160]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|   85 | Tensor<[1, 1024, 196]> self = ?,<br>List[int] size = [1, 1024, 14, 14]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   86 | Tensor<[1, 1024, 2560]> self = ?,<br>List[int] size = [1024, 2560]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|   87 | Tensor<[1, 1024, 256]> self = ?,<br>List[int] size = [1, 1024, 16, 16]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|   88 | Tensor<[1, 1024, 256]> self = ?,<br>List[int] size = [1024, 256]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   89 | Tensor<[1, 1024, 5, 32]> self = ?,<br>List[int] size = [1, 1024, 160]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|   90 | Tensor<[1, 1024, 640]> self = ?,<br>List[int] size = [1, -1, 8, 80]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|   91 | Tensor<[1, 1024, 640]> self = ?,<br>List[int] size = [1, 32, 32, 640]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|   92 | Tensor<[1, 1024, 640]> self = ?,<br>List[int] size = [1024, 640]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|   93 | Tensor<[1, 1024, 7, 7]> self = ?,<br>List[int] size = [1, 1024, 49]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|   94 | Tensor<[1, 1024, 8, 80]> self = ?,<br>List[int] size = [1, -1, 640]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|   95 | Tensor<[1, 1024]> self = ?,<br>List[int] size = [1, 1, 1024]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   96 | Tensor<[1, 1024]> self = ?,<br>List[int] size = [1, 1024, 1, 1]                | None     | N/A                 | N/A          | N/A               | N/A                |
|   97 | Tensor<[1, 1024]> self = ?,<br>List[int] size = [1024]                         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|   98 | Tensor<[1, 10]> self = ?,<br>List[int] size = [-1, 10]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|   99 | Tensor<[1, 10]> self = ?,<br>List[int] size = [10]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  100 | Tensor<[1, 12, 1, 10]> self = ?,<br>List[int] size = [12, 1, 10]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  101 | Tensor<[1, 12, 1, 1]> self = ?,<br>List[int] size = [12, 1, 1]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  102 | Tensor<[1, 12, 1, 24]> self = ?,<br>List[int] size = [12, 1, 24]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  103 | Tensor<[1, 12, 1, 2]> self = ?,<br>List[int] size = [12, 1, 2]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  104 | Tensor<[1, 12, 1, 46]> self = ?,<br>List[int] size = [12, 1, 46]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  105 | Tensor<[1, 12, 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  106 | Tensor<[1, 12, 1, 64]> self = ?,<br>List[int] size = [12, 1, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  107 | Tensor<[1, 12, 1, s0 + 1]> self = ?,<br>List[int] size = [12, 1, <s0 + 1>]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  108 | Tensor<[1, 12, 1, s10 + 1]> self = ?,<br>List[int] size = [12, 1, <s10 + 1>]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  109 | Tensor<[1, 12, 10, 10]> self = ?,<br>List[int] size = [12, 10, 10]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  110 | Tensor<[1, 12, 10, 64]> self = ?,<br>List[int] size = [12, 10, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  111 | Tensor<[1, 12, 12, 12]> self = ?,<br>List[int] size = [12, 12, 12]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  112 | Tensor<[1, 12, 12, 64]> self = ?,<br>List[int] size = [12, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  113 | Tensor<[1, 12, 128]> self = ?,<br>List[int] size = [12, 128]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  114 | Tensor<[1, 12, 14, 14]> self = ?,<br>List[int] size = [12, 14, 14]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  115 | Tensor<[1, 12, 14, 64]> self = ?,<br>List[int] size = [12, 14, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  116 | Tensor<[1, 12, 16, 16]> self = ?,<br>List[int] size = [12, 16, 16]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  117 | Tensor<[1, 12, 16, 64]> self = ?,<br>List[int] size = [12, 16, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  118 | Tensor<[1, 12, 197, 197]> self = ?,<br>List[int] size = [12, 197, 197]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  119 | Tensor<[1, 12, 197, 64]> self = ?,<br>List[int] size = [12, 197, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  120 | Tensor<[1, 12, 2, 64]> self = ?,<br>List[int] size = [12, -1, 64]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  121 | Tensor<[1, 12, 2, 64]> self = ?,<br>List[int] size = [12, 2, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  122 | Tensor<[1, 12, 201, 201]> self = ?,<br>List[int] size = [12, 201, 201]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  123 | Tensor<[1, 12, 201, 64]> self = ?,<br>List[int] size = [12, 201, 64]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  124 | Tensor<[1, 12, 24, 24]> self = ?,<br>List[int] size = [12, 24, 24]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  125 | Tensor<[1, 12, 24, 64]> self = ?,<br>List[int] size = [12, -1, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  126 | Tensor<[1, 12, 25, 25]> self = ?,<br>List[int] size = [12, 25, 25]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  127 | Tensor<[1, 12, 25, 64]> self = ?,<br>List[int] size = [12, 25, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  128 | Tensor<[1, 12, 3072]> self = ?,<br>List[int] size = [12, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  129 | Tensor<[1, 12, 45, 45]> self = ?,<br>List[int] size = [12, 45, 45]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  130 | Tensor<[1, 12, 45, 64]> self = ?,<br>List[int] size = [12, 45, 64]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  131 | Tensor<[1, 12, 46, 64]> self = ?,<br>List[int] size = [12, 46, 64]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  132 | Tensor<[1, 12, 50, 64]> self = ?,<br>List[int] size = [12, -1, 64]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  133 | Tensor<[1, 12, 50, 64]> self = ?,<br>List[int] size = [12, 50, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  134 | Tensor<[1, 12, 64, 10]> self = ?,<br>List[int] size = [12, 64, 10]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  135 | Tensor<[1, 12, 64, 12]> self = ?,<br>List[int] size = [12, 64, 12]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  136 | Tensor<[1, 12, 64, 14]> self = ?,<br>List[int] size = [12, 64, 14]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  137 | Tensor<[1, 12, 64, 16]> self = ?,<br>List[int] size = [12, 64, 16]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  138 | Tensor<[1, 12, 64, 197]> self = ?,<br>List[int] size = [12, 64, 197]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  139 | Tensor<[1, 12, 64, 1]> self = ?,<br>List[int] size = [12, 64, 1]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  140 | Tensor<[1, 12, 64, 201]> self = ?,<br>List[int] size = [12, 64, 201]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  141 | Tensor<[1, 12, 64, 25]> self = ?,<br>List[int] size = [12, 64, 25]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  142 | Tensor<[1, 12, 64, 2]> self = ?,<br>List[int] size = [12, 64, 2]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  143 | Tensor<[1, 12, 64, 45]> self = ?,<br>List[int] size = [12, 64, 45]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  144 | Tensor<[1, 12, 64, 46]> self = ?,<br>List[int] size = [12, 64, 46]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  145 | Tensor<[1, 12, 64, 7]> self = ?,<br>List[int] size = [12, 64, 7]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  146 | Tensor<[1, 12, 64, 9]> self = ?,<br>List[int] size = [12, 64, 9]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  147 | Tensor<[1, 12, 64, s0 + 1]> self = ?,<br>List[int] size = [12, 64, <s0 + 1>]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  148 | Tensor<[1, 12, 64, s10 + 1]> self = ?,<br>List[int] size = [12, 64, <s10 + 1>] | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  149 | Tensor<[1, 12, 7, 64]> self = ?,<br>List[int] size = [12, 7, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  150 | Tensor<[1, 12, 7, 7]> self = ?,<br>List[int] size = [12, 7, 7]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  151 | Tensor<[1, 12, 768]> self = ?,<br>List[int] size = [1, 12, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  152 | Tensor<[1, 12, 768]> self = ?,<br>List[int] size = [12, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  153 | Tensor<[1, 12, 9, 64]> self = ?,<br>List[int] size = [12, 9, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  154 | Tensor<[1, 12, 9, 9]> self = ?,<br>List[int] size = [12, 9, 9]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  155 | Tensor<[1, 12, s0 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  156 | Tensor<[1, 12, s0 + 1, 64]> self = ?,<br>List[int] size = [12, <s0 + 1>, 64]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  157 | Tensor<[1, 12, s10 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]        | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  158 | Tensor<[1, 12, s10 + 1, 64]> self = ?,<br>List[int] size = [12, <s10 + 1>, 64] | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  159 | Tensor<[1, 12, s2 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  160 | Tensor<[1, 12, s4 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  161 | Tensor<[1, 12, s6 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  162 | Tensor<[1, 12, s8 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  163 | Tensor<[1, 1200, 1280]> self = ?,<br>List[int] size = [1200, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  164 | Tensor<[1, 1200, 320]> self = ?,<br>List[int] size = [1, 1200, 5, 64]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  165 | Tensor<[1, 1200, 320]> self = ?,<br>List[int] size = [1, 30, 40, -1]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  166 | Tensor<[1, 1200, 320]> self = ?,<br>List[int] size = [1200, 320]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  167 | Tensor<[1, 1200, 5, 64]> self = ?,<br>List[int] size = [1, 1200, 320]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  168 | Tensor<[1, 128, 128, 128]> self = ?,<br>List[int] size = [1, 128, 16384]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  169 | Tensor<[1, 128, 128, 32]> self = ?,<br>List[int] size = [1, 16384, 32]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  170 | Tensor<[1, 128, 15, 20]> self = ?,<br>List[int] size = [1, 128, 300]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  171 | Tensor<[1, 128, 16384]> self = ?,<br>List[int] size = [1, 128, 128, 128]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  172 | Tensor<[1, 128, 4800]> self = ?,<br>List[int] size = [1, 128, 60, 80]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  173 | Tensor<[1, 128, 60, 80]> self = ?,<br>List[int] size = [1, 128, 4800]          | None     | N/A                 | N/A          | N/A               | N/A                |
|  174 | Tensor<[1, 1280, 1, 1]> self = ?,<br>List[int] size = [1, 1280]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  175 | Tensor<[1, 1280, 1200]> self = ?,<br>List[int] size = [1, 1280, 30, 40]        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  176 | Tensor<[1, 1280, 30, 40]> self = ?,<br>List[int] size = [1, 1280, 1200]        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  177 | Tensor<[1, 1280, 37, 37]> self = ?,<br>List[int] size = [1, 1280, 1369]        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  178 | Tensor<[1, 1280]> self = ?,<br>List[int] size = [1, 1280, 1, 1]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  179 | Tensor<[1, 128]> self = ?,<br>List[int] size = [128]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  180 | Tensor<[1, 12]> self = ?,<br>List[int] size = [-1, 2]                          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  181 | Tensor<[1, 12]> self = ?,<br>List[int] size = [12]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  182 | Tensor<[1, 1370, 1280]> self = ?,<br>List[int] size = [1370, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  183 | Tensor<[1, 1370, 5120]> self = ?,<br>List[int] size = [1370, 5120]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  184 | Tensor<[1, 14, 128]> self = ?,<br>List[int] size = [14, 128]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  185 | Tensor<[1, 14, 14, 1024]> self = ?,<br>List[int] size = [196, 1024]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  186 | Tensor<[1, 14, 14, 1536]> self = ?,<br>List[int] size = [196, 1536]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  187 | Tensor<[1, 14, 14, 2048]> self = ?,<br>List[int] size = [196, 2048]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  188 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] size = [1, 2, 7, 2, 7, 384]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  189 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] size = [196, 384]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  190 | Tensor<[1, 14, 14, 512]> self = ?,<br>List[int] size = [1, 2, 7, 2, 7, 512]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  191 | Tensor<[1, 14, 14, 512]> self = ?,<br>List[int] size = [196, 512]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  192 | Tensor<[1, 14, 14, 768]> self = ?,<br>List[int] size = [196, 768]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  193 | Tensor<[1, 14, 3072]> self = ?,<br>List[int] size = [14, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  194 | Tensor<[1, 14, 768]> self = ?,<br>List[int] size = [1, 14, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  195 | Tensor<[1, 14, 768]> self = ?,<br>List[int] size = [14, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  196 | Tensor<[1, 1445, 192]> self = ?,<br>List[int] size = [1, 1445, 3, 64]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  197 | Tensor<[1, 1445, 192]> self = ?,<br>List[int] size = [1445, 192]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  198 | Tensor<[1, 1445, 3, 64]> self = ?,<br>List[int] size = [1, 1445, 192]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  199 | Tensor<[1, 1445, 768]> self = ?,<br>List[int] size = [1445, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  200 | Tensor<[1, 15, 1024]> self = ?,<br>List[int] size = [15, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  201 | Tensor<[1, 15, 15, 12]> self = ?,<br>List[int] size = [-1, 12]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  202 | Tensor<[1, 15, 15, 16]> self = ?,<br>List[int] size = [-1, 16]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  203 | Tensor<[1, 15, 15, 24]> self = ?,<br>List[int] size = [-1, 24]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  204 | Tensor<[1, 15, 15, 2]> self = ?,<br>List[int] size = [225, 2]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  205 | Tensor<[1, 15, 15, 32]> self = ?,<br>List[int] size = [-1, 32]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  206 | Tensor<[1, 15, 15, 3]> self = ?,<br>List[int] size = [-1, 3]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  207 | Tensor<[1, 15, 15, 4]> self = ?,<br>List[int] size = [-1, 4]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  208 | Tensor<[1, 15, 15, 512]> self = ?,<br>List[int] size = [225, 512]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  209 | Tensor<[1, 15, 15, 6]> self = ?,<br>List[int] size = [-1, 6]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  210 | Tensor<[1, 15, 15, 8]> self = ?,<br>List[int] size = [-1, 8]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  211 | Tensor<[1, 15, 384]> self = ?,<br>List[int] size = [1, -1, 6, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  212 | Tensor<[1, 15, 384]> self = ?,<br>List[int] size = [15, 384]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  213 | Tensor<[1, 15, 512]> self = ?,<br>List[int] size = [15, 512]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  214 | Tensor<[1, 15, 6, 64]> self = ?,<br>List[int] size = [1, -1, 384]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  215 | Tensor<[1, 1500, 12, 64]> self = ?,<br>List[int] size = [1, 1500, 768]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  216 | Tensor<[1, 1500, 3072]> self = ?,<br>List[int] size = [1500, 3072]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  217 | Tensor<[1, 1500, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  218 | Tensor<[1, 1500, 768]> self = ?,<br>List[int] size = [1, 1500, 12, 64]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  219 | Tensor<[1, 1500, 768]> self = ?,<br>List[int] size = [1500, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  220 | Tensor<[1, 1512, 1, 1]> self = ?,<br>List[int] size = [1, 1512]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  221 | Tensor<[1, 1536, 1, 1]> self = ?,<br>List[int] size = [1, 1536]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  222 | Tensor<[1, 1536]> self = ?,<br>List[int] size = [1, 1536, 1, 1]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  223 | Tensor<[1, 15]> self = ?,<br>List[int] size = [-1, 15]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  224 | Tensor<[1, 16, 1, 10]> self = ?,<br>List[int] size = [16, 1, 10]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  225 | Tensor<[1, 16, 1, 1]> self = ?,<br>List[int] size = [1, -1, 4, 1, 1]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  226 | Tensor<[1, 16, 1, 1]> self = ?,<br>List[int] size = [16, 1, 1]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  227 | Tensor<[1, 16, 1, 2]> self = ?,<br>List[int] size = [16, 1, 2]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  228 | Tensor<[1, 16, 1, 60]> self = ?,<br>List[int] size = [16, 1, 60]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  229 | Tensor<[1, 16, 1, 64]> self = ?,<br>List[int] size = [16, -1, 64]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  230 | Tensor<[1, 16, 1, 64]> self = ?,<br>List[int] size = [16, 1, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  231 | Tensor<[1, 16, 1, 6]> self = ?,<br>List[int] size = [16, 1, 6]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  232 | Tensor<[1, 16, 1, s0 + 1]> self = ?,<br>List[int] size = [16, 1, <s0 + 1>]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  233 | Tensor<[1, 16, 1, s10 + 1]> self = ?,<br>List[int] size = [16, 1, <s10 + 1>]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  234 | Tensor<[1, 16, 10, 10]> self = ?,<br>List[int] size = [16, 10, 10]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  235 | Tensor<[1, 16, 10, 64]> self = ?,<br>List[int] size = [16, 10, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  236 | Tensor<[1, 16, 12, 64]> self = ?,<br>List[int] size = [1, -1, 768]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  237 | Tensor<[1, 16, 128, 9]> self = ?,<br>List[int] size = [16, 128, 9]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  238 | Tensor<[1, 16, 16, 1024]> self = ?,<br>List[int] size = [256, 1024]            | None     | N/A                 | N/A          | N/A               | N/A                |
|  239 | Tensor<[1, 16, 16, 1280]> self = ?,<br>List[int] size = [1, 256, 1280]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  240 | Tensor<[1, 16, 16, 1536]> self = ?,<br>List[int] size = [256, 1536]            | None     | N/A                 | N/A          | N/A               | N/A                |
|  241 | Tensor<[1, 16, 16, 2048]> self = ?,<br>List[int] size = [256, 2048]            | None     | N/A                 | N/A          | N/A               | N/A                |
|  242 | Tensor<[1, 16, 16, 256]> self = ?,<br>List[int] size = [1, 256, 256]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  243 | Tensor<[1, 16, 16, 384]> self = ?,<br>List[int] size = [1, 2, 8, 2, 8, 384]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  244 | Tensor<[1, 16, 16, 384]> self = ?,<br>List[int] size = [256, 384]              | None     | N/A                 | N/A          | N/A               | N/A                |
|  245 | Tensor<[1, 16, 16, 512]> self = ?,<br>List[int] size = [1, 2, 8, 2, 8, 512]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  246 | Tensor<[1, 16, 16, 512]> self = ?,<br>List[int] size = [256, 512]              | None     | N/A                 | N/A          | N/A               | N/A                |
|  247 | Tensor<[1, 16, 16, 768]> self = ?,<br>List[int] size = [256, 768]              | None     | N/A                 | N/A          | N/A               | N/A                |
|  248 | Tensor<[1, 16, 19, 19]> self = ?,<br>List[int] size = [16, 19, 19]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  249 | Tensor<[1, 16, 19, 64]> self = ?,<br>List[int] size = [16, -1, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  250 | Tensor<[1, 16, 197, 197]> self = ?,<br>List[int] size = [16, 197, 197]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  251 | Tensor<[1, 16, 197, 64]> self = ?,<br>List[int] size = [16, 197, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  252 | Tensor<[1, 16, 2, 64]> self = ?,<br>List[int] size = [16, 2, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  253 | Tensor<[1, 16, 256, 256]> self = ?,<br>List[int] size = [16, 256, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  254 | Tensor<[1, 16, 256, 64]> self = ?,<br>List[int] size = [16, 256, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  255 | Tensor<[1, 16, 3, 3]> self = ?,<br>List[int] size = [1, -1, 4, 3, 3]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  256 | Tensor<[1, 16, 3072]> self = ?,<br>List[int] size = [16, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  257 | Tensor<[1, 16, 32, 32]> self = ?,<br>List[int] size = [16, 32, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  258 | Tensor<[1, 16, 32, 96]> self = ?,<br>List[int] size = [16, 32, 96]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  259 | Tensor<[1, 16, 32]> self = ?,<br>List[int] size = [16, 1, 32]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  260 | Tensor<[1, 16, 38, 38]> self = ?,<br>List[int] size = [1, -1, 4, 38, 38]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  261 | Tensor<[1, 16, 5, 5]> self = ?,<br>List[int] size = [16, 5, 5]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  262 | Tensor<[1, 16, 5, 64]> self = ?,<br>List[int] size = [16, 5, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  263 | Tensor<[1, 16, 59, 59]> self = ?,<br>List[int] size = [16, 59, 59]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  264 | Tensor<[1, 16, 59, 64]> self = ?,<br>List[int] size = [16, -1, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  265 | Tensor<[1, 16, 6, 49, 49]> self = ?,<br>List[int] size = [-1, 6, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  266 | Tensor<[1, 16, 6, 64, 64]> self = ?,<br>List[int] size = [-1, 6, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  267 | Tensor<[1, 16, 6, 64]> self = ?,<br>List[int] size = [16, 6, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  268 | Tensor<[1, 16, 60, 64]> self = ?,<br>List[int] size = [16, -1, 64]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  269 | Tensor<[1, 16, 64, 10]> self = ?,<br>List[int] size = [16, 64, 10]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  270 | Tensor<[1, 16, 64, 197]> self = ?,<br>List[int] size = [16, 64, 197]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  271 | Tensor<[1, 16, 64, 1]> self = ?,<br>List[int] size = [16, 64, 1]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  272 | Tensor<[1, 16, 64, 256]> self = ?,<br>List[int] size = [16, 64, 256]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  273 | Tensor<[1, 16, 64, 2]> self = ?,<br>List[int] size = [16, 64, 2]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  274 | Tensor<[1, 16, 64, 5]> self = ?,<br>List[int] size = [16, 64, 5]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  275 | Tensor<[1, 16, 64, 6]> self = ?,<br>List[int] size = [16, 64, 6]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  276 | Tensor<[1, 16, 64, 9]> self = ?,<br>List[int] size = [16, 64, 9]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  277 | Tensor<[1, 16, 64, s0 + 1]> self = ?,<br>List[int] size = [16, 64, <s0 + 1>]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  278 | Tensor<[1, 16, 64, s10 + 1]> self = ?,<br>List[int] size = [16, 64, <s10 + 1>] | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  279 | Tensor<[1, 16, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  280 | Tensor<[1, 16, 768]> self = ?,<br>List[int] size = [16, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  281 | Tensor<[1, 16, 8, 49, 49]> self = ?,<br>List[int] size = [-1, 8, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  282 | Tensor<[1, 16, 8, 64, 64]> self = ?,<br>List[int] size = [-1, 8, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  283 | Tensor<[1, 16, 9, 128]> self = ?,<br>List[int] size = [16, 9, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  284 | Tensor<[1, 16, 9, 64]> self = ?,<br>List[int] size = [16, 9, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  285 | Tensor<[1, 16, 9, 9]> self = ?,<br>List[int] size = [16, 9, 9]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  286 | Tensor<[1, 16, 96, 32]> self = ?,<br>List[int] size = [16, 96, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  287 | Tensor<[1, 16, s0 + 1, 64]> self = ?,<br>List[int] size = [16, <s0 + 1>, 64]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  288 | Tensor<[1, 16, s10 + 1, 64]> self = ?,<br>List[int] size = [16, -1, 64]        | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  289 | Tensor<[1, 16, s10 + 1, 64]> self = ?,<br>List[int] size = [16, <s10 + 1>, 64] | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  290 | Tensor<[1, 160, 1024]> self = ?,<br>List[int] size = [1, 160, 32, 32]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  291 | Tensor<[1, 160, 16, 16]> self = ?,<br>List[int] size = [1, 160, 256]           | None     | N/A                 | N/A          | N/A               | N/A                |
|  292 | Tensor<[1, 160, 256]> self = ?,<br>List[int] size = [1, 160, 16, 16]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  293 | Tensor<[1, 160, 32, 32]> self = ?,<br>List[int] size = [1, 160, 1024]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  294 | Tensor<[1, 160]> self = ?,<br>List[int] size = [160]                           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  295 | Tensor<[1, 16384, 1, 32]> self = ?,<br>List[int] size = [1, 16384, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  296 | Tensor<[1, 16384, 128]> self = ?,<br>List[int] size = [16384, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  297 | Tensor<[1, 16384, 256]> self = ?,<br>List[int] size = [1, 1, 16384, 256]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  298 | Tensor<[1, 16384, 256]> self = ?,<br>List[int] size = [16384, 256]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  299 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [1, 1, 16384, 32]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  300 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [1, 128, 128, -1]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  301 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [1, 16384, 1, 32]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  302 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [16384, 32]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  303 | Tensor<[1, 1664, 1, 1]> self = ?,<br>List[int] size = [1, 1664]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  304 | Tensor<[1, 16]> self = ?,<br>List[int] size = [1, 1, 1, 16]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  305 | Tensor<[1, 19, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  306 | Tensor<[1, 19, 1024]> self = ?,<br>List[int] size = [1, 19, 16, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  307 | Tensor<[1, 19, 1024]> self = ?,<br>List[int] size = [19, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  308 | Tensor<[1, 19, 256008]> self = ?,<br>List[int] size = [-1, 256008]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  309 | Tensor<[1, 19, 4096]> self = ?,<br>List[int] size = [19, 4096]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  310 | Tensor<[1, 192, 32, 42]> self = ?,<br>List[int] size = [1, 192, 1344]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  311 | Tensor<[1, 192, 4150]> self = ?,<br>List[int] size = [1, 192, 50, 83]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  312 | Tensor<[1, 1920, 1, 1]> self = ?,<br>List[int] size = [1, 1920]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  313 | Tensor<[1, 19200, 1, 64]> self = ?,<br>List[int] size = [1, 19200, 64]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  314 | Tensor<[1, 19200, 256]> self = ?,<br>List[int] size = [19200, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  315 | Tensor<[1, 19200, 300]> self = ?,<br>List[int] size = [1, 1, 19200, 300]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  316 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [1, 1, 19200, 64]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  317 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [1, 120, 160, -1]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  318 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [1, 19200, 1, 64]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  319 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [19200, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  320 | Tensor<[1, 196, 3072]> self = ?,<br>List[int] size = [196, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  321 | Tensor<[1, 196, 768]> self = ?,<br>List[int] size = [196, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  322 | Tensor<[1, 196]> self = ?,<br>List[int] size = [196]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  323 | Tensor<[1, 197, 1024]> self = ?,<br>List[int] size = [1, 197, 16, 64]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  324 | Tensor<[1, 197, 1024]> self = ?,<br>List[int] size = [197, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  325 | Tensor<[1, 197, 12, 64]> self = ?,<br>List[int] size = [1, 197, 768]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  326 | Tensor<[1, 197, 16, 64]> self = ?,<br>List[int] size = [1, 197, 1024]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  327 | Tensor<[1, 197, 3072]> self = ?,<br>List[int] size = [197, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  328 | Tensor<[1, 197, 4096]> self = ?,<br>List[int] size = [197, 4096]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  329 | Tensor<[1, 197, 768]> self = ?,<br>List[int] size = [1, 197, 12, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  330 | Tensor<[1, 197, 768]> self = ?,<br>List[int] size = [197, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  331 | Tensor<[1, 19]> self = ?,<br>List[int] size = [-1, 19]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  332 | Tensor<[1, 19]> self = ?,<br>List[int] size = [-1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  333 | Tensor<[1, 1]> self = ?,<br>List[int] size = [-1, 1]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  334 | Tensor<[1, 1]> self = ?,<br>List[int] size = [-1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  335 | Tensor<[1, 1]> self = ?,<br>List[int] size = [1]                               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  336 | Tensor<[1, 2, 256, 32]> self = ?,<br>List[int] size = [2, 256, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  337 | Tensor<[1, 2, 300, 64]> self = ?,<br>List[int] size = [2, 300, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  338 | Tensor<[1, 2, 32, 256]> self = ?,<br>List[int] size = [2, 32, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  339 | Tensor<[1, 2, 4096, 256]> self = ?,<br>List[int] size = [2, 4096, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  340 | Tensor<[1, 2, 4096, 32]> self = ?,<br>List[int] size = [2, 4096, 32]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  341 | Tensor<[1, 2, 4800, 300]> self = ?,<br>List[int] size = [2, 4800, 300]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  342 | Tensor<[1, 2, 4800, 64]> self = ?,<br>List[int] size = [2, 4800, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  343 | Tensor<[1, 2, 64, 300]> self = ?,<br>List[int] size = [2, 64, 300]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  344 | Tensor<[1, 201, 12, 64]> self = ?,<br>List[int] size = [1, 201, 768]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  345 | Tensor<[1, 201, 3072]> self = ?,<br>List[int] size = [201, 3072]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  346 | Tensor<[1, 201, 768]> self = ?,<br>List[int] size = [1, 201, 12, 64]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  347 | Tensor<[1, 201, 768]> self = ?,<br>List[int] size = [201, 768]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  348 | Tensor<[1, 2016, 1, 1]> self = ?,<br>List[int] size = [1, 2016]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  349 | Tensor<[1, 2048, 1, 1]> self = ?,<br>List[int] size = [1, 2048]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  350 | Tensor<[1, 2048, 1280]> self = ?,<br>List[int] size = [1, 2048, 8, 160]        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  351 | Tensor<[1, 2048, 15, 20]> self = ?,<br>List[int] size = [1, 2048, 300]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  352 | Tensor<[1, 2048, 256]> self = ?,<br>List[int] size = [1, 2048, 8, 32]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  353 | Tensor<[1, 2048, 300]> self = ?,<br>List[int] size = [1, 2048, 15, 20]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  354 | Tensor<[1, 2048, 768]> self = ?,<br>List[int] size = [-1, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  355 | Tensor<[1, 2048, 768]> self = ?,<br>List[int] size = [2048, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  356 | Tensor<[1, 2048, 8, 96]> self = ?,<br>List[int] size = [1, 2048, 768]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  357 | Tensor<[1, 2048]> self = ?,<br>List[int] size = [1, 1, 2048]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  358 | Tensor<[1, 2048]> self = ?,<br>List[int] size = [1, 2048, 1, 1]                | None     | N/A                 | N/A          | N/A               | N/A                |
|  359 | Tensor<[1, 2048]> self = ?,<br>List[int] size = [2048]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  360 | Tensor<[1, 21843]> self = ?,<br>List[int] size = [21843]                       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  361 | Tensor<[1, 2208, 1, 1]> self = ?,<br>List[int] size = [1, 2208]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  362 | Tensor<[1, 23, 40, 64, 2]> self = ?,<br>List[int] size = [1, 23, 40, 128]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  363 | Tensor<[1, 23, 40]> self = ?,<br>List[int] size = [1, 920]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  364 | Tensor<[1, 24, 1, 1]> self = ?,<br>List[int] size = [1, -1, 4, 1, 1]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  365 | Tensor<[1, 24, 10, 10]> self = ?,<br>List[int] size = [1, -1, 4, 10, 10]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  366 | Tensor<[1, 24, 19, 19]> self = ?,<br>List[int] size = [1, -1, 4, 19, 19]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  367 | Tensor<[1, 24, 2, 2]> self = ?,<br>List[int] size = [1, -1, 4, 2, 2]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  368 | Tensor<[1, 24, 20, 20]> self = ?,<br>List[int] size = [1, -1, 4, 20, 20]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  369 | Tensor<[1, 24, 3, 3]> self = ?,<br>List[int] size = [1, -1, 4, 3, 3]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  370 | Tensor<[1, 24, 3072]> self = ?,<br>List[int] size = [24, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  371 | Tensor<[1, 24, 32, 49]> self = ?,<br>List[int] size = [24, 32, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  372 | Tensor<[1, 24, 32, 64]> self = ?,<br>List[int] size = [24, 32, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  373 | Tensor<[1, 24, 49, 32]> self = ?,<br>List[int] size = [24, 49, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  374 | Tensor<[1, 24, 49, 49]> self = ?,<br>List[int] size = [24, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  375 | Tensor<[1, 24, 5, 5]> self = ?,<br>List[int] size = [1, -1, 4, 5, 5]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  376 | Tensor<[1, 24, 64, 32]> self = ?,<br>List[int] size = [24, 64, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  377 | Tensor<[1, 24, 64, 64]> self = ?,<br>List[int] size = [24, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  378 | Tensor<[1, 24, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  379 | Tensor<[1, 24, 768]> self = ?,<br>List[int] size = [1, 24, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  380 | Tensor<[1, 24, 768]> self = ?,<br>List[int] size = [24, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  381 | Tensor<[1, 25, 12, 64]> self = ?,<br>List[int] size = [1, 25, 768]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  382 | Tensor<[1, 25, 3072]> self = ?,<br>List[int] size = [25, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  383 | Tensor<[1, 25, 768]> self = ?,<br>List[int] size = [1, 25, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  384 | Tensor<[1, 25, 768]> self = ?,<br>List[int] size = [25, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  385 | Tensor<[1, 2520, 1, 1]> self = ?,<br>List[int] size = [1, 2520]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  386 | Tensor<[1, 256, 1, 32]> self = ?,<br>List[int] size = [1, 256, 32]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  387 | Tensor<[1, 256, 1024]> self = ?,<br>List[int] size = [1, 256, 16, 64]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  388 | Tensor<[1, 256, 1024]> self = ?,<br>List[int] size = [1, 256, 32, 32]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  389 | Tensor<[1, 256, 1024]> self = ?,<br>List[int] size = [256, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  390 | Tensor<[1, 256, 120, 160]> self = ?,<br>List[int] size = [1, 256, 19200]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  391 | Tensor<[1, 256, 128, 128]> self = ?,<br>List[int] size = [1, 256, 16384]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  392 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [1, -1, 8, 160]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  393 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [1, 16, 16, 1280]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  394 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [1, 256, 8, 160]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  395 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [256, 1280]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  396 | Tensor<[1, 256, 16, 16]> self = ?,<br>List[int] size = [1, 256, 256]           | None     | N/A                 | N/A          | N/A               | N/A                |
|  397 | Tensor<[1, 256, 16, 64]> self = ?,<br>List[int] size = [1, 256, 1024]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  398 | Tensor<[1, 256, 160]> self = ?,<br>List[int] size = [1, 256, 5, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  399 | Tensor<[1, 256, 160]> self = ?,<br>List[int] size = [256, 160]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  400 | Tensor<[1, 256, 16384]> self = ?,<br>List[int] size = [1, 256, 128, 128]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  401 | Tensor<[1, 256, 19200]> self = ?,<br>List[int] size = [1, 256, 120, 160]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  402 | Tensor<[1, 256, 2, 32]> self = ?,<br>List[int] size = [1, 256, 64]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  403 | Tensor<[1, 256, 23, 40]> self = ?,<br>List[int] size = [1, 256, 920]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  404 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [1, 16, 16, -1]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  405 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [1, 256, 16, 16]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  406 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [1, 256, 8, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  407 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [256, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  408 | Tensor<[1, 256, 32, 32]> self = ?,<br>List[int] size = [1, 256, 1024]          | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  409 | Tensor<[1, 256, 32]> self = ?,<br>List[int] size = [1, 1, 256, 32]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  410 | Tensor<[1, 256, 32]> self = ?,<br>List[int] size = [1, 256, 1, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  411 | Tensor<[1, 256, 32]> self = ?,<br>List[int] size = [256, 32]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  412 | Tensor<[1, 256, 4096]> self = ?,<br>List[int] size = [1, 256, 64, 64]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  413 | Tensor<[1, 256, 4096]> self = ?,<br>List[int] size = [256, 4096]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  414 | Tensor<[1, 256, 5, 32]> self = ?,<br>List[int] size = [1, 256, 160]            | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  415 | Tensor<[1, 256, 5120]> self = ?,<br>List[int] size = [256, 5120]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  416 | Tensor<[1, 256, 512]> self = ?,<br>List[int] size = [256, 512]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  417 | Tensor<[1, 256, 64, 64]> self = ?,<br>List[int] size = [1, 256, 4096]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  418 | Tensor<[1, 256, 64]> self = ?,<br>List[int] size = [1, 256, 2, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  419 | Tensor<[1, 256, 64]> self = ?,<br>List[int] size = [256, 64]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  420 | Tensor<[1, 256, 768]> self = ?,<br>List[int] size = [1, 16, 16, 16, 16, 3]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  421 | Tensor<[1, 256, 768]> self = ?,<br>List[int] size = [1, 256, 8, 96]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  422 | Tensor<[1, 256, 768]> self = ?,<br>List[int] size = [256, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  423 | Tensor<[1, 256, 8, 160]> self = ?,<br>List[int] size = [1, -1, 1280]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  424 | Tensor<[1, 256, 8, 160]> self = ?,<br>List[int] size = [1, 256, 1280]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  425 | Tensor<[1, 256, 8, 32]> self = ?,<br>List[int] size = [1, 256, 256]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  426 | Tensor<[1, 256]> self = ?,<br>List[int] size = [1, 1, 256]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  427 | Tensor<[1, 256]> self = ?,<br>List[int] size = [256]                           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  428 | Tensor<[1, 25]> self = ?,<br>List[int] size = [1, 25]                          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  429 | Tensor<[1, 28, 28, 1024]> self = ?,<br>List[int] size = [784, 1024]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  430 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] size = [1, 4, 7, 4, 7, 192]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  431 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] size = [784, 192]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  432 | Tensor<[1, 28, 28, 256]> self = ?,<br>List[int] size = [1, 4, 7, 4, 7, 256]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  433 | Tensor<[1, 28, 28, 256]> self = ?,<br>List[int] size = [784, 256]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  434 | Tensor<[1, 28, 28, 384]> self = ?,<br>List[int] size = [784, 384]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  435 | Tensor<[1, 28, 28, 512]> self = ?,<br>List[int] size = [784, 512]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  436 | Tensor<[1, 28, 28, 768]> self = ?,<br>List[int] size = [784, 768]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  437 | Tensor<[1, 3, 1445, 1445]> self = ?,<br>List[int] size = [3, 1445, 1445]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  438 | Tensor<[1, 3, 1445, 64]> self = ?,<br>List[int] size = [3, 1445, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  439 | Tensor<[1, 3, 256, 256]> self = ?,<br>List[int] size = [1, 3, 16, 16, 16, 16]  | None     | N/A                 | N/A          | N/A               | N/A                |
|  440 | Tensor<[1, 3, 64, 1445]> self = ?,<br>List[int] size = [3, 64, 1445]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  441 | Tensor<[1, 300, 128]> self = ?,<br>List[int] size = [1, 300, 2, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  442 | Tensor<[1, 300, 128]> self = ?,<br>List[int] size = [300, 128]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  443 | Tensor<[1, 300, 2048]> self = ?,<br>List[int] size = [300, 2048]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  444 | Tensor<[1, 300, 320]> self = ?,<br>List[int] size = [1, 300, 5, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  445 | Tensor<[1, 300, 320]> self = ?,<br>List[int] size = [300, 320]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  446 | Tensor<[1, 300, 512]> self = ?,<br>List[int] size = [1, 15, 20, -1]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  447 | Tensor<[1, 300, 512]> self = ?,<br>List[int] size = [1, 300, 8, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  448 | Tensor<[1, 300, 512]> self = ?,<br>List[int] size = [300, 512]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  449 | Tensor<[1, 300, 64]> self = ?,<br>List[int] size = [1, 300, 1, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  450 | Tensor<[1, 300, 64]> self = ?,<br>List[int] size = [300, 64]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  451 | Tensor<[1, 300, 8, 64]> self = ?,<br>List[int] size = [1, 300, 512]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  452 | Tensor<[1, 3024, 1, 1]> self = ?,<br>List[int] size = [1, 3024]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  453 | Tensor<[1, 3072]> self = ?,<br>List[int] size = [1, 1, 3072]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  454 | Tensor<[1, 3072]> self = ?,<br>List[int] size = [3072]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  455 | Tensor<[1, 32, 128, 128]> self = ?,<br>List[int] size = [1, 32, 16384]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  456 | Tensor<[1, 32, 1536]> self = ?,<br>List[int] size = [32, 1536]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  457 | Tensor<[1, 32, 16, 16]> self = ?,<br>List[int] size = [1, 32, 256]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  458 | Tensor<[1, 32, 16384]> self = ?,<br>List[int] size = [1, 32, 128, 128]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  459 | Tensor<[1, 32, 256]> self = ?,<br>List[int] size = [1, 1, 32, 256]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  460 | Tensor<[1, 32, 256]> self = ?,<br>List[int] size = [1, 32, 16, 16]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  461 | Tensor<[1, 32, 32, 1024]> self = ?,<br>List[int] size = [1024, 1024]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  462 | Tensor<[1, 32, 32, 160]> self = ?,<br>List[int] size = [1, 1024, 160]          | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  463 | Tensor<[1, 32, 32, 192]> self = ?,<br>List[int] size = [1, 4, 8, 4, 8, 192]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  464 | Tensor<[1, 32, 32, 192]> self = ?,<br>List[int] size = [1024, 192]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  465 | Tensor<[1, 32, 32, 256]> self = ?,<br>List[int] size = [1, 4, 8, 4, 8, 256]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  466 | Tensor<[1, 32, 32, 256]> self = ?,<br>List[int] size = [1024, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  467 | Tensor<[1, 32, 32, 384]> self = ?,<br>List[int] size = [1024, 384]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  468 | Tensor<[1, 32, 32, 49]> self = ?,<br>List[int] size = [32, 32, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  469 | Tensor<[1, 32, 32, 512]> self = ?,<br>List[int] size = [1024, 512]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  470 | Tensor<[1, 32, 32, 640]> self = ?,<br>List[int] size = [1, 1024, 640]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  471 | Tensor<[1, 32, 32, 64]> self = ?,<br>List[int] size = [32, 32, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  472 | Tensor<[1, 32, 32, 768]> self = ?,<br>List[int] size = [1024, 768]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  473 | Tensor<[1, 32, 4608]> self = ?,<br>List[int] size = [1, 32, 16, 3, 96]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  474 | Tensor<[1, 32, 49, 32]> self = ?,<br>List[int] size = [32, 49, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  475 | Tensor<[1, 32, 49, 49]> self = ?,<br>List[int] size = [32, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  476 | Tensor<[1, 32, 6144]> self = ?,<br>List[int] size = [32, 6144]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  477 | Tensor<[1, 32, 64, 32]> self = ?,<br>List[int] size = [32, 64, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  478 | Tensor<[1, 32, 64, 64]> self = ?,<br>List[int] size = [32, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  479 | Tensor<[1, 320, 1200]> self = ?,<br>List[int] size = [1, 320, 30, 40]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  480 | Tensor<[1, 320, 15, 20]> self = ?,<br>List[int] size = [1, 320, 300]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  481 | Tensor<[1, 320, 30, 40]> self = ?,<br>List[int] size = [1, 320, 1200]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  482 | Tensor<[1, 32128]> self = ?,<br>List[int] size = [1, 1, 32128]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  483 | Tensor<[1, 32]> self = ?,<br>List[int] size = [32]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  484 | Tensor<[1, 36, 100, 136]> self = ?,<br>List[int] size = [1, -1, 4, 100, 136]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  485 | Tensor<[1, 36, 13, 17]> self = ?,<br>List[int] size = [1, -1, 4, 13, 17]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  486 | Tensor<[1, 36, 25, 34]> self = ?,<br>List[int] size = [1, -1, 4, 25, 34]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  487 | Tensor<[1, 36, 50, 68]> self = ?,<br>List[int] size = [1, -1, 4, 50, 68]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  488 | Tensor<[1, 36, 7, 9]> self = ?,<br>List[int] size = [1, -1, 4, 7, 9]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  489 | Tensor<[1, 364, 1, 1]> self = ?,<br>List[int] size = [1, -1, 91, 1, 1]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  490 | Tensor<[1, 364, 3, 3]> self = ?,<br>List[int] size = [1, -1, 91, 3, 3]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  491 | Tensor<[1, 364, 38, 38]> self = ?,<br>List[int] size = [1, -1, 91, 38, 38]     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  492 | Tensor<[1, 3712, 1, 1]> self = ?,<br>List[int] size = [1, 3712]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  493 | Tensor<[1, 384]> self = ?,<br>List[int] size = [1, 1, 384]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  494 | Tensor<[1, 3]> self = ?,<br>List[int] size = [3]                               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  495 | Tensor<[1, 4, 12, 49, 49]> self = ?,<br>List[int] size = [-1, 12, 49, 49]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  496 | Tensor<[1, 4, 12, 64, 64]> self = ?,<br>List[int] size = [-1, 12, 64, 64]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  497 | Tensor<[1, 4, 12, 64]> self = ?,<br>List[int] size = [1, 4, 768]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  498 | Tensor<[1, 4, 16, 49, 49]> self = ?,<br>List[int] size = [-1, 16, 49, 49]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  499 | Tensor<[1, 4, 16, 64, 64]> self = ?,<br>List[int] size = [-1, 16, 64, 64]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  500 | Tensor<[1, 4, 3072]> self = ?,<br>List[int] size = [4, 3072]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  501 | Tensor<[1, 4, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  502 | Tensor<[1, 4, 768]> self = ?,<br>List[int] size = [1, 4, 12, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  503 | Tensor<[1, 4, 768]> self = ?,<br>List[int] size = [4, 768]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  504 | Tensor<[1, 400, 1, 1]> self = ?,<br>List[int] size = [1, 400]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  505 | Tensor<[1, 4096, 1280]> self = ?,<br>List[int] size = [4096, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  506 | Tensor<[1, 4096, 2, 32]> self = ?,<br>List[int] size = [1, 4096, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  507 | Tensor<[1, 4096, 256]> self = ?,<br>List[int] size = [4096, 256]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  508 | Tensor<[1, 4096, 320]> self = ?,<br>List[int] size = [1, -1, 8, 40]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  509 | Tensor<[1, 4096, 320]> self = ?,<br>List[int] size = [1, 64, 64, 320]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  510 | Tensor<[1, 4096, 320]> self = ?,<br>List[int] size = [4096, 320]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  511 | Tensor<[1, 4096, 64]> self = ?,<br>List[int] size = [1, 4096, 2, 32]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  512 | Tensor<[1, 4096, 64]> self = ?,<br>List[int] size = [1, 64, 64, -1]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  513 | Tensor<[1, 4096, 64]> self = ?,<br>List[int] size = [4096, 64]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  514 | Tensor<[1, 4096, 8, 40]> self = ?,<br>List[int] size = [1, -1, 320]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  515 | Tensor<[1, 4096]> self = ?,<br>List[int] size = [1, 1, 4096]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  516 | Tensor<[1, 4096]> self = ?,<br>List[int] size = [4096]                         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  517 | Tensor<[1, 440, 1, 1]> self = ?,<br>List[int] size = [1, 440]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  518 | Tensor<[1, 45, 12, 64]> self = ?,<br>List[int] size = [1, 45, 768]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  519 | Tensor<[1, 45, 3072]> self = ?,<br>List[int] size = [45, 3072]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  520 | Tensor<[1, 45, 768]> self = ?,<br>List[int] size = [-1, 45, 768]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  521 | Tensor<[1, 45, 768]> self = ?,<br>List[int] size = [1, 45, 12, 64]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  522 | Tensor<[1, 45, 768]> self = ?,<br>List[int] size = [45, 768]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  523 | Tensor<[1, 45]> self = ?,<br>List[int] size = [-1, 45]                         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  524 | Tensor<[1, 4800, 128]> self = ?,<br>List[int] size = [1, 4800, 2, 64]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  525 | Tensor<[1, 4800, 128]> self = ?,<br>List[int] size = [1, 60, 80, -1]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  526 | Tensor<[1, 4800, 128]> self = ?,<br>List[int] size = [4800, 128]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  527 | Tensor<[1, 4800, 2, 64]> self = ?,<br>List[int] size = [1, 4800, 128]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  528 | Tensor<[1, 4800, 512]> self = ?,<br>List[int] size = [4800, 512]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  529 | Tensor<[1, 49, 1024]> self = ?,<br>List[int] size = [1, 1, 1, 7, 7, 1024]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  530 | Tensor<[1, 49, 1024]> self = ?,<br>List[int] size = [49, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  531 | Tensor<[1, 49, 2304]> self = ?,<br>List[int] size = [1, 49, 3, 24, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  532 | Tensor<[1, 49, 3072]> self = ?,<br>List[int] size = [1, 49, 3, 32, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  533 | Tensor<[1, 49, 768]> self = ?,<br>List[int] size = [1, 1, 1, 7, 7, 768]        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  534 | Tensor<[1, 49, 768]> self = ?,<br>List[int] size = [49, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  535 | Tensor<[1, 4]> self = ?,<br>List[int] size = [-1, 4]                           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  536 | Tensor<[1, 5, 1, 16, 2]> self = ?,<br>List[int] size = [1, 5, 1, 32]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  537 | Tensor<[1, 5, 1024, 256]> self = ?,<br>List[int] size = [5, 1024, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  538 | Tensor<[1, 5, 1024, 32]> self = ?,<br>List[int] size = [5, 1024, 32]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  539 | Tensor<[1, 5, 1024]> self = ?,<br>List[int] size = [1, 5, 1024]                | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  540 | Tensor<[1, 5, 1024]> self = ?,<br>List[int] size = [5, 1024]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  541 | Tensor<[1, 5, 1200, 300]> self = ?,<br>List[int] size = [5, 1200, 300]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  542 | Tensor<[1, 5, 1200, 64]> self = ?,<br>List[int] size = [5, 1200, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  543 | Tensor<[1, 5, 16, 16, 2]> self = ?,<br>List[int] size = [1, 5, 16, 32]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  544 | Tensor<[1, 5, 16, 64]> self = ?,<br>List[int] size = [1, 5, 1024]              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  545 | Tensor<[1, 5, 256, 32]> self = ?,<br>List[int] size = [5, 256, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  546 | Tensor<[1, 5, 300, 64]> self = ?,<br>List[int] size = [5, 300, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  547 | Tensor<[1, 5, 3072]> self = ?,<br>List[int] size = [1, 5, 4, -1]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  548 | Tensor<[1, 5, 32, 256]> self = ?,<br>List[int] size = [5, 32, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  549 | Tensor<[1, 5, 4, 256]> self = ?,<br>List[int] size = [1, 5, 4, 4, 64]          | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  550 | Tensor<[1, 5, 4096]> self = ?,<br>List[int] size = [5, 4096]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  551 | Tensor<[1, 5, 64, 300]> self = ?,<br>List[int] size = [5, 64, 300]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  552 | Tensor<[1, 50, 1024]> self = ?,<br>List[int] size = [50, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  553 | Tensor<[1, 50, 12, 64]> self = ?,<br>List[int] size = [1, 50, 768]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  554 | Tensor<[1, 50, 3072]> self = ?,<br>List[int] size = [50, 3072]                 | None     | N/A                 | N/A          | N/A               | N/A                |
|  555 | Tensor<[1, 50, 4096]> self = ?,<br>List[int] size = [50, 4096]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  556 | Tensor<[1, 50, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  557 | Tensor<[1, 50, 768]> self = ?,<br>List[int] size = [1, 50, 12, 64]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  558 | Tensor<[1, 50, 768]> self = ?,<br>List[int] size = [50, 768]                   | None     | N/A                 | N/A          | N/A               | N/A                |
|  559 | Tensor<[1, 50257]> self = ?,<br>List[int] size = [1, 1, 50257]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  560 | Tensor<[1, 50272]> self = ?,<br>List[int] size = [1, 1, 50272]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  561 | Tensor<[1, 512, 1, 1]> self = ?,<br>List[int] size = [1, 512]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  562 | Tensor<[1, 512, 15, 20]> self = ?,<br>List[int] size = [1, 512, 300]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  563 | Tensor<[1, 512, 4800]> self = ?,<br>List[int] size = [1, 512, 60, 80]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  564 | Tensor<[1, 512, 60, 80]> self = ?,<br>List[int] size = [1, 512, 4800]          | None     | N/A                 | N/A          | N/A               | N/A                |
|  565 | Tensor<[1, 512, 7, 7]> self = ?,<br>List[int] size = [1, 25088]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  566 | Tensor<[1, 51200]> self = ?,<br>List[int] size = [1, 1, 51200]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  567 | Tensor<[1, 512]> self = ?,<br>List[int] size = [1, 1, 512]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  568 | Tensor<[1, 512]> self = ?,<br>List[int] size = [1, 512, 1, 1]                  | None     | N/A                 | N/A          | N/A               | N/A                |
|  569 | Tensor<[1, 512]> self = ?,<br>List[int] size = [512]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  570 | Tensor<[1, 51865]> self = ?,<br>List[int] size = [1, 1, 51865]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  571 | Tensor<[1, 546, 1, 1]> self = ?,<br>List[int] size = [1, -1, 91, 1, 1]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  572 | Tensor<[1, 546, 10, 10]> self = ?,<br>List[int] size = [1, -1, 91, 10, 10]     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  573 | Tensor<[1, 546, 19, 19]> self = ?,<br>List[int] size = [1, -1, 91, 19, 19]     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  574 | Tensor<[1, 546, 2, 2]> self = ?,<br>List[int] size = [1, -1, 91, 2, 2]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  575 | Tensor<[1, 546, 20, 20]> self = ?,<br>List[int] size = [1, -1, 91, 20, 20]     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  576 | Tensor<[1, 546, 3, 3]> self = ?,<br>List[int] size = [1, -1, 91, 3, 3]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  577 | Tensor<[1, 546, 5, 5]> self = ?,<br>List[int] size = [1, -1, 91, 5, 5]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  578 | Tensor<[1, 56, 56, 128]> self = ?,<br>List[int] size = [1, 8, 7, 8, 7, 128]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  579 | Tensor<[1, 56, 56, 128]> self = ?,<br>List[int] size = [3136, 128]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  580 | Tensor<[1, 56, 56, 384]> self = ?,<br>List[int] size = [3136, 384]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  581 | Tensor<[1, 56, 56, 512]> self = ?,<br>List[int] size = [3136, 512]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  582 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] size = [1, 8, 7, 8, 7, 96]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  583 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] size = [3136, 96]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  584 | Tensor<[1, 576, 1, 1]> self = ?,<br>List[int] size = [1, 576]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  585 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [-1, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  586 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  587 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [1, 59, 16, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  588 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [59, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  589 | Tensor<[1, 59, 512]> self = ?,<br>List[int] size = [59, 512]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  590 | Tensor<[1, 59]> self = ?,<br>List[int] size = [-1, 59]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  591 | Tensor<[1, 5]> self = ?,<br>List[int] size = [-1, 5]                           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  592 | Tensor<[1, 5]> self = ?,<br>List[int] size = [1, -1]                           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  593 | Tensor<[1, 6, 1, 15]> self = ?,<br>List[int] size = [6, 1, 15]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  594 | Tensor<[1, 6, 1, 17]> self = ?,<br>List[int] size = [6, 1, 17]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  595 | Tensor<[1, 6, 1, 1]> self = ?,<br>List[int] size = [6, 1, 1]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  596 | Tensor<[1, 6, 1, 2]> self = ?,<br>List[int] size = [6, 1, 2]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  597 | Tensor<[1, 6, 1, 64]> self = ?,<br>List[int] size = [6, 1, 64]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  598 | Tensor<[1, 6, 1, s0 + 1]> self = ?,<br>List[int] size = [6, 1, <s0 + 1>]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  599 | Tensor<[1, 6, 15, 15]> self = ?,<br>List[int] size = [6, 15, 15]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  600 | Tensor<[1, 6, 15, 64]> self = ?,<br>List[int] size = [6, 15, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  601 | Tensor<[1, 6, 17, 64]> self = ?,<br>List[int] size = [6, 17, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  602 | Tensor<[1, 6, 2, 64]> self = ?,<br>List[int] size = [6, 2, 64]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  603 | Tensor<[1, 6, 64, 15]> self = ?,<br>List[int] size = [6, 64, 15]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  604 | Tensor<[1, 6, 64, 17]> self = ?,<br>List[int] size = [6, 64, 17]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  605 | Tensor<[1, 6, 64, 1]> self = ?,<br>List[int] size = [6, 64, 1]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  606 | Tensor<[1, 6, 64, 2]> self = ?,<br>List[int] size = [6, 64, 2]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  607 | Tensor<[1, 6, 64, s0 + 1]> self = ?,<br>List[int] size = [6, 64, <s0 + 1>]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  608 | Tensor<[1, 6, s0 + 1, 64]> self = ?,<br>List[int] size = [6, <s0 + 1>, 64]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  609 | Tensor<[1, 64, 1024]> self = ?,<br>List[int] size = [1, 1, 1, 8, 8, 1024]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  610 | Tensor<[1, 64, 1024]> self = ?,<br>List[int] size = [64, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  611 | Tensor<[1, 64, 12, 12]> self = ?,<br>List[int] size = [1, 9216]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  612 | Tensor<[1, 64, 120, 160]> self = ?,<br>List[int] size = [1, 64, 19200]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  613 | Tensor<[1, 64, 1280]> self = ?,<br>List[int] size = [1, -1, 8, 160]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  614 | Tensor<[1, 64, 1280]> self = ?,<br>List[int] size = [1, 8, 8, 1280]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  615 | Tensor<[1, 64, 1280]> self = ?,<br>List[int] size = [64, 1280]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  616 | Tensor<[1, 64, 15, 20]> self = ?,<br>List[int] size = [1, 64, 300]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  617 | Tensor<[1, 64, 16, 16]> self = ?,<br>List[int] size = [1, 64, 256]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  618 | Tensor<[1, 64, 19200]> self = ?,<br>List[int] size = [1, 64, 120, 160]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  619 | Tensor<[1, 64, 2304]> self = ?,<br>List[int] size = [1, 64, 3, 24, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  620 | Tensor<[1, 64, 256]> self = ?,<br>List[int] size = [1, 64, 16, 16]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  621 | Tensor<[1, 64, 3, 49, 49]> self = ?,<br>List[int] size = [-1, 3, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  622 | Tensor<[1, 64, 3, 64, 64]> self = ?,<br>List[int] size = [-1, 3, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  623 | Tensor<[1, 64, 3072]> self = ?,<br>List[int] size = [1, 64, 3, 32, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  624 | Tensor<[1, 64, 4, 49, 49]> self = ?,<br>List[int] size = [-1, 4, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  625 | Tensor<[1, 64, 4, 64, 64]> self = ?,<br>List[int] size = [-1, 4, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  626 | Tensor<[1, 64, 4096]> self = ?,<br>List[int] size = [1, 64, 64, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  627 | Tensor<[1, 64, 5120]> self = ?,<br>List[int] size = [64, 5120]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  628 | Tensor<[1, 64, 64, 128]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 128]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  629 | Tensor<[1, 64, 64, 128]> self = ?,<br>List[int] size = [4096, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  630 | Tensor<[1, 64, 64, 320]> self = ?,<br>List[int] size = [1, 4096, 320]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  631 | Tensor<[1, 64, 64, 384]> self = ?,<br>List[int] size = [4096, 384]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  632 | Tensor<[1, 64, 64, 512]> self = ?,<br>List[int] size = [4096, 512]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  633 | Tensor<[1, 64, 64, 64]> self = ?,<br>List[int] size = [1, 4096, 64]            | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  634 | Tensor<[1, 64, 64, 64]> self = ?,<br>List[int] size = [1, 64, 4096]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  635 | Tensor<[1, 64, 64, 96]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 96]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  636 | Tensor<[1, 64, 64, 96]> self = ?,<br>List[int] size = [4096, 96]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  637 | Tensor<[1, 64, 64, 9]> self = ?,<br>List[int] size = [64, 64, 9]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  638 | Tensor<[1, 64, 768]> self = ?,<br>List[int] size = [1, 1, 1, 8, 8, 768]        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  639 | Tensor<[1, 64, 768]> self = ?,<br>List[int] size = [64, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  640 | Tensor<[1, 64, 8, 160]> self = ?,<br>List[int] size = [1, -1, 1280]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  641 | Tensor<[1, 64, 9, 64]> self = ?,<br>List[int] size = [64, 9, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  642 | Tensor<[1, 64, 9, 9]> self = ?,<br>List[int] size = [64, 9, 9]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  643 | Tensor<[1, 640, 1024]> self = ?,<br>List[int] size = [1, 640, 32, 32]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  644 | Tensor<[1, 640, 32, 32]> self = ?,<br>List[int] size = [1, 640, 1024]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  645 | Tensor<[1, 640]> self = ?,<br>List[int] size = [640]                           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  646 | Tensor<[1, 64]> self = ?,<br>List[int] size = [64]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  647 | Tensor<[1, 672, 1, 1]> self = ?,<br>List[int] size = [1, 672]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  648 | Tensor<[1, 6]> self = ?,<br>List[int] size = [1, -1]                           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  649 | Tensor<[1, 7, 12, 64]> self = ?,<br>List[int] size = [1, 7, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  650 | Tensor<[1, 7, 18176]> self = ?,<br>List[int] size = [7, 18176]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  651 | Tensor<[1, 7, 3072]> self = ?,<br>List[int] size = [-1, 3072]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  652 | Tensor<[1, 7, 4544]> self = ?,<br>List[int] size = [7, 4544]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  653 | Tensor<[1, 7, 4672]> self = ?,<br>List[int] size = [1, 7, 73, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  654 | Tensor<[1, 7, 7, 1024]> self = ?,<br>List[int] size = [1, 1, 7, 1, 7, 1024]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  655 | Tensor<[1, 7, 7, 1024]> self = ?,<br>List[int] size = [49, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  656 | Tensor<[1, 7, 7, 1536]> self = ?,<br>List[int] size = [49, 1536]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  657 | Tensor<[1, 7, 7, 2048]> self = ?,<br>List[int] size = [49, 2048]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  658 | Tensor<[1, 7, 7, 3072]> self = ?,<br>List[int] size = [49, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  659 | Tensor<[1, 7, 7, 4096]> self = ?,<br>List[int] size = [49, 4096]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  660 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] size = [1, 1, 7, 1, 7, 768]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  661 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] size = [49, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  662 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [-1, 7, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  663 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [-1, 768]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  664 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [1, 7, 12, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  665 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [7, 768]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  666 | Tensor<[1, 71, 64, 7]> self = ?,<br>List[int] size = [71, 64, 7]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  667 | Tensor<[1, 71, 7, 64]> self = ?,<br>List[int] size = [1, 71, 7, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  668 | Tensor<[1, 71, 7, 64]> self = ?,<br>List[int] size = [71, 7, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  669 | Tensor<[1, 71, 7, 7]> self = ?,<br>List[int] size = [71, 7, 7]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  670 | Tensor<[1, 7392, 1, 1]> self = ?,<br>List[int] size = [1, 7392]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  671 | Tensor<[1, 768, 1, 1]> self = ?,<br>List[int] size = [1, 768]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  672 | Tensor<[1, 768, 12, 16]> self = ?,<br>List[int] size = [1, 768, 192]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  673 | Tensor<[1, 768, 14, 14]> self = ?,<br>List[int] size = [1, 768, 196]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  674 | Tensor<[1, 768, 144]> self = ?,<br>List[int] size = [1, 768, 12, 12]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  675 | Tensor<[1, 768, 196]> self = ?,<br>List[int] size = [1, 768, 14, 14]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  676 | Tensor<[1, 768, 196]> self = ?,<br>List[int] size = [768, 196]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  677 | Tensor<[1, 768, 384]> self = ?,<br>List[int] size = [768, 384]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  678 | Tensor<[1, 768, 49]> self = ?,<br>List[int] size = [1, 768, 7, 7]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  679 | Tensor<[1, 768, 7, 7]> self = ?,<br>List[int] size = [1, 768, 49]              | None     | N/A                 | N/A          | N/A               | N/A                |
|  680 | Tensor<[1, 768]> self = ?,<br>List[int] size = [1, 1, 768]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  681 | Tensor<[1, 768]> self = ?,<br>List[int] size = [768]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  682 | Tensor<[1, 784, 1, 1]> self = ?,<br>List[int] size = [1, 784]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  683 | Tensor<[1, 784]> self = ?,<br>List[int] size = [784]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  684 | Tensor<[1, 7]> self = ?,<br>List[int] size = [-1, 7]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  685 | Tensor<[1, 7]> self = ?,<br>List[int] size = [1, -1]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  686 | Tensor<[1, 8, 1, 10]> self = ?,<br>List[int] size = [8, 1, 10]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  687 | Tensor<[1, 8, 1, 1]> self = ?,<br>List[int] size = [8, 1, 1]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  688 | Tensor<[1, 8, 1, 2]> self = ?,<br>List[int] size = [8, 1, 2]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  689 | Tensor<[1, 8, 1, 64]> self = ?,<br>List[int] size = [8, 1, 64]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  690 | Tensor<[1, 8, 1, 920]> self = ?,<br>List[int] size = [8, 1, 920]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  691 | Tensor<[1, 8, 1, s0 + 1]> self = ?,<br>List[int] size = [8, 1, <s0 + 1>]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  692 | Tensor<[1, 8, 10, 10]> self = ?,<br>List[int] size = [8, 10, 10]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  693 | Tensor<[1, 8, 10, 64]> self = ?,<br>List[int] size = [8, 10, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  694 | Tensor<[1, 8, 2, 64]> self = ?,<br>List[int] size = [8, 2, 64]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  695 | Tensor<[1, 8, 2048, 160]> self = ?,<br>List[int] size = [8, 2048, 160]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  696 | Tensor<[1, 8, 2048, 256]> self = ?,<br>List[int] size = [8, 2048, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  697 | Tensor<[1, 8, 2048, 32]> self = ?,<br>List[int] size = [8, 2048, 32]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  698 | Tensor<[1, 8, 256, 160]> self = ?,<br>List[int] size = [8, 256, 160]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  699 | Tensor<[1, 8, 256, 2048]> self = ?,<br>List[int] size = [8, 256, 2048]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  700 | Tensor<[1, 8, 256, 256]> self = ?,<br>List[int] size = [8, 256, 256]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  701 | Tensor<[1, 8, 256, 32]> self = ?,<br>List[int] size = [8, 256, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  702 | Tensor<[1, 8, 256, 96]> self = ?,<br>List[int] size = [8, 256, 96]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  703 | Tensor<[1, 8, 300, 300]> self = ?,<br>List[int] size = [8, 300, 300]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  704 | Tensor<[1, 8, 300, 64]> self = ?,<br>List[int] size = [8, 300, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  705 | Tensor<[1, 8, 32, 2048]> self = ?,<br>List[int] size = [8, 32, 2048]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  706 | Tensor<[1, 8, 32, 256]> self = ?,<br>List[int] size = [8, 32, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  707 | Tensor<[1, 8, 64, 10]> self = ?,<br>List[int] size = [8, 64, 10]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  708 | Tensor<[1, 8, 64, 1]> self = ?,<br>List[int] size = [8, 64, 1]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  709 | Tensor<[1, 8, 64, 2]> self = ?,<br>List[int] size = [8, 64, 2]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  710 | Tensor<[1, 8, 64, 300]> self = ?,<br>List[int] size = [8, 64, 300]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  711 | Tensor<[1, 8, 64, s0 + 1]> self = ?,<br>List[int] size = [8, 64, <s0 + 1>]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  712 | Tensor<[1, 8, 8, 1024]> self = ?,<br>List[int] size = [1, 1, 8, 1, 8, 1024]    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  713 | Tensor<[1, 8, 8, 1024]> self = ?,<br>List[int] size = [64, 1024]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  714 | Tensor<[1, 8, 8, 1280]> self = ?,<br>List[int] size = [1, 64, 1280]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  715 | Tensor<[1, 8, 8, 1536]> self = ?,<br>List[int] size = [64, 1536]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  716 | Tensor<[1, 8, 8, 2048]> self = ?,<br>List[int] size = [64, 2048]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  717 | Tensor<[1, 8, 8, 3072]> self = ?,<br>List[int] size = [64, 3072]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  718 | Tensor<[1, 8, 8, 4096]> self = ?,<br>List[int] size = [64, 4096]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  719 | Tensor<[1, 8, 8, 768]> self = ?,<br>List[int] size = [1, 1, 8, 1, 8, 768]      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  720 | Tensor<[1, 8, 8, 768]> self = ?,<br>List[int] size = [64, 768]                 | None     | N/A                 | N/A          | N/A               | N/A                |
|  721 | Tensor<[1, 8, s0 + 1, 64]> self = ?,<br>List[int] size = [8, <s0 + 1>, 64]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  722 | Tensor<[1, 819, 100, 136]> self = ?,<br>List[int] size = [1, -1, 91, 100, 136] | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  723 | Tensor<[1, 819, 13, 17]> self = ?,<br>List[int] size = [1, -1, 91, 13, 17]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  724 | Tensor<[1, 819, 25, 34]> self = ?,<br>List[int] size = [1, -1, 91, 25, 34]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  725 | Tensor<[1, 819, 50, 68]> self = ?,<br>List[int] size = [1, -1, 91, 50, 68]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  726 | Tensor<[1, 819, 7, 9]> self = ?,<br>List[int] size = [1, -1, 91, 7, 9]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  727 | Tensor<[1, 888, 1, 1]> self = ?,<br>List[int] size = [1, 888]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  728 | Tensor<[1, 8]> self = ?,<br>List[int] size = [-1, 2]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  729 | Tensor<[1, 9, 1024]> self = ?,<br>List[int] size = [1, 9, 16, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  730 | Tensor<[1, 9, 1024]> self = ?,<br>List[int] size = [9, 1024]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  731 | Tensor<[1, 9, 1280]> self = ?,<br>List[int] size = [1, -1, 8, 160]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  732 | Tensor<[1, 9, 128]> self = ?,<br>List[int] size = [9, 128]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  733 | Tensor<[1, 9, 16384]> self = ?,<br>List[int] size = [9, 16384]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  734 | Tensor<[1, 9, 2048]> self = ?,<br>List[int] size = [1, 9, 16, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  735 | Tensor<[1, 9, 2048]> self = ?,<br>List[int] size = [9, 2048]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  736 | Tensor<[1, 9, 3072]> self = ?,<br>List[int] size = [9, 3072]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  737 | Tensor<[1, 9, 320]> self = ?,<br>List[int] size = [1, -1, 8, 40]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  738 | Tensor<[1, 9, 4096]> self = ?,<br>List[int] size = [1, 9, 64, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  739 | Tensor<[1, 9, 4096]> self = ?,<br>List[int] size = [9, 4096]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  740 | Tensor<[1, 9, 640]> self = ?,<br>List[int] size = [1, -1, 8, 80]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  741 | Tensor<[1, 9, 768]> self = ?,<br>List[int] size = [1, 9, 12, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  742 | Tensor<[1, 9, 768]> self = ?,<br>List[int] size = [9, 768]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  743 | Tensor<[1, 9, 8192]> self = ?,<br>List[int] size = [9, 8192]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  744 | Tensor<[1, 912, 1, 1]> self = ?,<br>List[int] size = [1, 912]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  745 | Tensor<[1, 920]> self = ?,<br>List[int] size = [1, 1, 1, 920]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  746 | Tensor<[1, 9216]> self = ?,<br>List[int] size = [1, 64, 12, 12]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  747 | Tensor<[1, 960, 1, 1]> self = ?,<br>List[int] size = [1, 960]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  748 | Tensor<[1, s0, 1280]> self = ?,<br>List[int] size = [<s0>, 1280]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  749 | Tensor<[1, s0, 256]> self = ?,<br>List[int] size = [<s0>, 256]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  750 | Tensor<[1, s0, 80]> self = ?,<br>List[int] size = [<s0>, 80]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  751 | Tensor<[1, s10 + 1]> self = ?,<br>List[int] size = [1, -1]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  752 | Tensor<[10, 1024]> self = ?,<br>List[int] size = [1, 10, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  753 | Tensor<[10, 2048]> self = ?,<br>List[int] size = [1, 10, 2048]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  754 | Tensor<[10, 250002]> self = ?,<br>List[int] size = [1, 10, 250002]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  755 | Tensor<[10, 3072]> self = ?,<br>List[int] size = [1, 10, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  756 | Tensor<[10, 4096]> self = ?,<br>List[int] size = [1, 10, 4096]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  757 | Tensor<[10, 512]> self = ?,<br>List[int] size = [1, 10, 512]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  758 | Tensor<[10, 768]> self = ?,<br>List[int] size = [1, 10, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  759 | Tensor<[100, 1, 2048]> self = ?,<br>List[int] size = [100, 2048]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  760 | Tensor<[100, 1, 256]> self = ?,<br>List[int] size = [100, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  761 | Tensor<[100, 1, 256]> self = ?,<br>List[int] size = [100, 8, 32]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  762 | Tensor<[100, 12]> self = ?,<br>List[int] size = [-1, 2]                        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  763 | Tensor<[100, 192]> self = ?,<br>List[int] size = [1, 100, 192]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  764 | Tensor<[100, 2048]> self = ?,<br>List[int] size = [100, 1, 2048]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  765 | Tensor<[100, 256]> self = ?,<br>List[int] size = [100, 1, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  766 | Tensor<[100, 4]> self = ?,<br>List[int] size = [1, 100, 4]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  767 | Tensor<[100, 8, 32]> self = ?,<br>List[int] size = [100, 256]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  768 | Tensor<[100, 92]> self = ?,<br>List[int] size = [1, 100, 92]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  769 | Tensor<[100]> self = ?,<br>List[int] size = [-1, 1]                            | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  770 | Tensor<[1024, 1024]> self = ?,<br>List[int] size = [1, 32, 32, 1024]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  771 | Tensor<[1024, 160]> self = ?,<br>List[int] size = [1, 1024, 160]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  772 | Tensor<[1024, 192]> self = ?,<br>List[int] size = [1, 32, 32, 192]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  773 | Tensor<[1024, 192]> self = ?,<br>List[int] size = [16, 64, 192]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  774 | Tensor<[1024, 256]> self = ?,<br>List[int] size = [1, 1024, 256]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  775 | Tensor<[1024, 256]> self = ?,<br>List[int] size = [1, 32, 32, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  776 | Tensor<[1024, 256]> self = ?,<br>List[int] size = [16, 64, 256]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  777 | Tensor<[1024, 5120]> self = ?,<br>List[int] size = [1, 1024, 5120]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  778 | Tensor<[1024, 576]> self = ?,<br>List[int] size = [16, 64, 576]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  779 | Tensor<[1024, 640]> self = ?,<br>List[int] size = [1, 1024, 640]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  780 | Tensor<[1024, 768]> self = ?,<br>List[int] size = [1, 32, 32, 768]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  781 | Tensor<[1024, 768]> self = ?,<br>List[int] size = [16, 64, 768]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  782 | Tensor<[1024]> self = ?,<br>List[int] size = [1, -1, 1, 1]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  783 | Tensor<[10]> self = ?,<br>List[int] size = [-1, 1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  784 | Tensor<[10]> self = ?,<br>List[int] size = [1, -1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  785 | Tensor<[12, 1, 10]> self = ?,<br>List[int] size = [1, 12, 1, 10]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  786 | Tensor<[12, 1, 1]> self = ?,<br>List[int] size = [1, 12, 1, 1]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  787 | Tensor<[12, 1, 24]> self = ?,<br>List[int] size = [1, 12, 1, 24]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  788 | Tensor<[12, 1, 2]> self = ?,<br>List[int] size = [1, 12, 1, 2]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  789 | Tensor<[12, 1, 46]> self = ?,<br>List[int] size = [1, 12, 1, 46]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  790 | Tensor<[12, 1, 64]> self = ?,<br>List[int] size = [1, 12, 1, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  791 | Tensor<[12, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 12, 1, <s0 + 1>]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  792 | Tensor<[12, 1, s10 + 1]> self = ?,<br>List[int] size = [1, 12, 1, <s10 + 1>]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  793 | Tensor<[12, 10, 10]> self = ?,<br>List[int] size = [1, 12, 10, 10]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  794 | Tensor<[12, 10, 64]> self = ?,<br>List[int] size = [1, 12, 10, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  795 | Tensor<[12, 12, 12]> self = ?,<br>List[int] size = [1, 12, 12, 12]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  796 | Tensor<[12, 12, 64]> self = ?,<br>List[int] size = [1, 12, 12, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  797 | Tensor<[12, 14, 14]> self = ?,<br>List[int] size = [1, 12, 14, 14]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  798 | Tensor<[12, 14, 64]> self = ?,<br>List[int] size = [1, 12, 14, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  799 | Tensor<[12, 16, 16]> self = ?,<br>List[int] size = [1, 12, 16, 16]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  800 | Tensor<[12, 16, 64]> self = ?,<br>List[int] size = [1, 12, 16, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  801 | Tensor<[12, 197, 197]> self = ?,<br>List[int] size = [1, 12, 197, 197]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  802 | Tensor<[12, 197, 64]> self = ?,<br>List[int] size = [1, 12, 197, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  803 | Tensor<[12, 201, 201]> self = ?,<br>List[int] size = [1, 12, 201, 201]         | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  804 | Tensor<[12, 201, 64]> self = ?,<br>List[int] size = [1, 12, 201, 64]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  805 | Tensor<[12, 24, 24]> self = ?,<br>List[int] size = [1, 12, 24, 24]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  806 | Tensor<[12, 24, 24]> self = ?,<br>List[int] size = [12, 24, 24]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  807 | Tensor<[12, 24, 64]> self = ?,<br>List[int] size = [1, 12, 24, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  808 | Tensor<[12, 24, 64]> self = ?,<br>List[int] size = [12, -1, 64]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  809 | Tensor<[12, 25, 25]> self = ?,<br>List[int] size = [1, 12, 25, 25]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  810 | Tensor<[12, 25, 64]> self = ?,<br>List[int] size = [1, 12, 25, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  811 | Tensor<[12, 2]> self = ?,<br>List[int] size = [1, 12, 2]                       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  812 | Tensor<[12, 3072]> self = ?,<br>List[int] size = [1, 12, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  813 | Tensor<[12, 45, 45]> self = ?,<br>List[int] size = [1, 12, 45, 45]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  814 | Tensor<[12, 45, 64]> self = ?,<br>List[int] size = [1, 12, 45, 64]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  815 | Tensor<[12, 50, 64]> self = ?,<br>List[int] size = [1, 12, 50, 64]             | None     | N/A                 | N/A          | N/A               | N/A                |
|  816 | Tensor<[12, 64, 197]> self = ?,<br>List[int] size = [1, 12, 64, 197]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  817 | Tensor<[12, 7, 64]> self = ?,<br>List[int] size = [1, 12, 7, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  818 | Tensor<[12, 7, 7]> self = ?,<br>List[int] size = [1, 12, 7, 7]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  819 | Tensor<[12, 768]> self = ?,<br>List[int] size = [1, 12, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  820 | Tensor<[12, 9, 64]> self = ?,<br>List[int] size = [1, 12, 9, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  821 | Tensor<[12, 9, 9]> self = ?,<br>List[int] size = [1, 12, 9, 9]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  822 | Tensor<[1200, 1280]> self = ?,<br>List[int] size = [1, 1200, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  823 | Tensor<[1200, 320]> self = ?,<br>List[int] size = [1, 1200, 320]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  824 | Tensor<[128, 49, 32]> self = ?,<br>List[int] size = [16, 8, 49, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  825 | Tensor<[128, 49, 49]> self = ?,<br>List[int] size = [16, 8, 49, 49]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  826 | Tensor<[128, 64, 32]> self = ?,<br>List[int] size = [16, 8, 64, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  827 | Tensor<[128, 64, 64]> self = ?,<br>List[int] size = [16, 8, 64, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  828 | Tensor<[128]> self = ?,<br>List[int] size = [1, -1, 1, 1]                      | Done     | N/A                 | N/A          | N/A               | N/A                |
|  829 | Tensor<[12]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  830 | Tensor<[13600, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                    | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  831 | Tensor<[13600, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                    | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  832 | Tensor<[136]> self = ?,<br>List[int] size = [1, -1]                            | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  833 | Tensor<[1370, 1, 1280]> self = ?,<br>List[int] size = [1370, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  834 | Tensor<[1370, 1, 1280]> self = ?,<br>List[int] size = [1370, 16, 80]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  835 | Tensor<[1370, 1, 16, 80]> self = ?,<br>List[int] size = [1370, 1280]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  836 | Tensor<[1370, 1, 3840]> self = ?,<br>List[int] size = [1370, 1, 3, 1280]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  837 | Tensor<[1370, 1280]> self = ?,<br>List[int] size = [1, 1370, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  838 | Tensor<[1370, 1280]> self = ?,<br>List[int] size = [1370, 1, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  839 | Tensor<[1370, 3840]> self = ?,<br>List[int] size = [1370, 1, 3840]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  840 | Tensor<[1370, 5120]> self = ?,<br>List[int] size = [1, 1370, 5120]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  841 | Tensor<[13]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  842 | Tensor<[14, 14]> self = ?,<br>List[int] size = [2, 7, 2, 7]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  843 | Tensor<[14, 2048]> self = ?,<br>List[int] size = [2, 7, 2048]                  | None     | N/A                 | N/A          | N/A               | N/A                |
|  844 | Tensor<[14, 2]> self = ?,<br>List[int] size = [1, 14, 2]                       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  845 | Tensor<[14, 3072]> self = ?,<br>List[int] size = [1, 14, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  846 | Tensor<[14, 512]> self = ?,<br>List[int] size = [2, 7, 512]                    | None     | N/A                 | N/A          | N/A               | N/A                |
|  847 | Tensor<[14, 768]> self = ?,<br>List[int] size = [1, 14, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  848 | Tensor<[1444, 8]> self = ?,<br>List[int] size = [-1, 2]                        | Done     | N/A                 | N/A          | N/A               | N/A                |
|  849 | Tensor<[1445, 192]> self = ?,<br>List[int] size = [1, 1445, 192]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  850 | Tensor<[1445, 768]> self = ?,<br>List[int] size = [1, 1445, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  851 | Tensor<[15, 1024]> self = ?,<br>List[int] size = [1, 15, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  852 | Tensor<[15, 384]> self = ?,<br>List[int] size = [1, 15, 384]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  853 | Tensor<[15, 512]> self = ?,<br>List[int] size = [1, 15, 512]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  854 | Tensor<[1500, 3072]> self = ?,<br>List[int] size = [1, 1500, 3072]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  855 | Tensor<[1500, 768]> self = ?,<br>List[int] size = [1, 1500, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  856 | Tensor<[16, 1, 10]> self = ?,<br>List[int] size = [1, 16, 1, 10]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  857 | Tensor<[16, 1, 1]> self = ?,<br>List[int] size = [1, 16, 1, 1]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  858 | Tensor<[16, 1, 2]> self = ?,<br>List[int] size = [1, 16, 1, 2]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  859 | Tensor<[16, 1, 60]> self = ?,<br>List[int] size = [1, 16, 1, 60]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  860 | Tensor<[16, 1, 64]> self = ?,<br>List[int] size = [1, 16, 1, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  861 | Tensor<[16, 1, 6]> self = ?,<br>List[int] size = [1, 16, 1, 6]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  862 | Tensor<[16, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 16, 1, <s0 + 1>]     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  863 | Tensor<[16, 1, s10 + 1]> self = ?,<br>List[int] size = [1, 16, 1, <s10 + 1>]   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  864 | Tensor<[16, 10, 10]> self = ?,<br>List[int] size = [1, 16, 10, 10]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  865 | Tensor<[16, 10, 64]> self = ?,<br>List[int] size = [1, 16, 10, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  866 | Tensor<[16, 1370, 80]> self = ?,<br>List[int] size = [1, 16, 1370, 80]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  867 | Tensor<[16, 16]> self = ?,<br>List[int] size = [2, 8, 2, 8]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
|  868 | Tensor<[16, 19, 19]> self = ?,<br>List[int] size = [1, 16, 19, 19]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  869 | Tensor<[16, 19, 64]> self = ?,<br>List[int] size = [1, 16, 19, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  870 | Tensor<[16, 197, 197]> self = ?,<br>List[int] size = [1, 16, 197, 197]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  871 | Tensor<[16, 197, 64]> self = ?,<br>List[int] size = [1, 16, 197, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  872 | Tensor<[16, 256, 256]> self = ?,<br>List[int] size = [1, 16, 256, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  873 | Tensor<[16, 256, 64]> self = ?,<br>List[int] size = [1, 16, 256, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  874 | Tensor<[16, 3072]> self = ?,<br>List[int] size = [1, 16, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  875 | Tensor<[16, 32, 32]> self = ?,<br>List[int] size = [1, 16, 32, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  876 | Tensor<[16, 32, 96]> self = ?,<br>List[int] size = [1, 16, 32, 96]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  877 | Tensor<[16, 49, 192]> self = ?,<br>List[int] size = [1, 4, 4, 7, 7, 192]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  878 | Tensor<[16, 49, 192]> self = ?,<br>List[int] size = [784, 192]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  879 | Tensor<[16, 49, 256]> self = ?,<br>List[int] size = [1, 4, 4, 7, 7, 256]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  880 | Tensor<[16, 49, 256]> self = ?,<br>List[int] size = [784, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  881 | Tensor<[16, 49, 576]> self = ?,<br>List[int] size = [16, 49, 3, 6, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  882 | Tensor<[16, 49, 768]> self = ?,<br>List[int] size = [16, 49, 3, 8, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  883 | Tensor<[16, 5, 5]> self = ?,<br>List[int] size = [1, 16, 5, 5]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  884 | Tensor<[16, 5, 64]> self = ?,<br>List[int] size = [1, 16, 5, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  885 | Tensor<[16, 50, 64]> self = ?,<br>List[int] size = [1, 16, 50, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  886 | Tensor<[16, 59, 59]> self = ?,<br>List[int] size = [1, 16, 59, 59]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  887 | Tensor<[16, 59, 64]> self = ?,<br>List[int] size = [1, 16, 59, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  888 | Tensor<[16, 6, 49, 49]> self = ?,<br>List[int] size = [1, 16, 6, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  889 | Tensor<[16, 6, 49, 49]> self = ?,<br>List[int] size = [96, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  890 | Tensor<[16, 6, 64, 64]> self = ?,<br>List[int] size = [1, 16, 6, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  891 | Tensor<[16, 6, 64, 64]> self = ?,<br>List[int] size = [96, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  892 | Tensor<[16, 64, 192]> self = ?,<br>List[int] size = [1, 4, 4, 8, 8, 192]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  893 | Tensor<[16, 64, 192]> self = ?,<br>List[int] size = [1024, 192]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  894 | Tensor<[16, 64, 197]> self = ?,<br>List[int] size = [1, 16, 64, 197]           | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  895 | Tensor<[16, 64, 256]> self = ?,<br>List[int] size = [1, 4, 4, 8, 8, 256]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  896 | Tensor<[16, 64, 256]> self = ?,<br>List[int] size = [1024, 256]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  897 | Tensor<[16, 64, 576]> self = ?,<br>List[int] size = [16, 64, 3, 6, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  898 | Tensor<[16, 64, 768]> self = ?,<br>List[int] size = [16, 64, 3, 8, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
|  899 | Tensor<[16, 7, 64]> self = ?,<br>List[int] size = [2, 8, 7, 64]                | None     | N/A                 | N/A          | N/A               | N/A                |
|  900 | Tensor<[16, 7, 7]> self = ?,<br>List[int] size = [2, 8, 7, 7]                  | None     | N/A                 | N/A          | N/A               | N/A                |
|  901 | Tensor<[16, 768]> self = ?,<br>List[int] size = [1, 16, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
|  902 | Tensor<[16, 8, 49, 49]> self = ?,<br>List[int] size = [1, 16, 8, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  903 | Tensor<[16, 8, 49, 49]> self = ?,<br>List[int] size = [128, 49, 49]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  904 | Tensor<[16, 8, 64, 64]> self = ?,<br>List[int] size = [1, 16, 8, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
|  905 | Tensor<[16, 8, 64, 64]> self = ?,<br>List[int] size = [128, 64, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  906 | Tensor<[16, 9, 128]> self = ?,<br>List[int] size = [1, 16, 9, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  907 | Tensor<[16, 9, 64]> self = ?,<br>List[int] size = [1, 16, 9, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  908 | Tensor<[16, 9, 9]> self = ?,<br>List[int] size = [1, 16, 9, 9]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  909 | Tensor<[16384, 128]> self = ?,<br>List[int] size = [1, 16384, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  910 | Tensor<[16384, 256]> self = ?,<br>List[int] size = [1, 16384, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  911 | Tensor<[16384, 32]> self = ?,<br>List[int] size = [1, 16384, 32]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  912 | Tensor<[16]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  913 | Tensor<[17]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  914 | Tensor<[19, 1024]> self = ?,<br>List[int] size = [1, 19, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  915 | Tensor<[19, 256008]> self = ?,<br>List[int] size = [1, 19, 256008]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  916 | Tensor<[19, 4096]> self = ?,<br>List[int] size = [1, 19, 4096]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  917 | Tensor<[192, 49, 32]> self = ?,<br>List[int] size = [64, 3, 49, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  918 | Tensor<[192, 49, 49]> self = ?,<br>List[int] size = [64, 3, 49, 49]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  919 | Tensor<[192, 64, 32]> self = ?,<br>List[int] size = [64, 3, 64, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  920 | Tensor<[192, 64, 64]> self = ?,<br>List[int] size = [64, 3, 64, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  921 | Tensor<[19200, 256]> self = ?,<br>List[int] size = [1, 19200, 256]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  922 | Tensor<[19200, 64]> self = ?,<br>List[int] size = [1, 19200, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  923 | Tensor<[192]> self = ?,<br>List[int] size = [1, 192, 1, 1]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  924 | Tensor<[196, 1152]> self = ?,<br>List[int] size = [4, 49, 1152]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  925 | Tensor<[196, 1536]> self = ?,<br>List[int] size = [1, 14, 14, 1536]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  926 | Tensor<[196, 1536]> self = ?,<br>List[int] size = [4, 49, 1536]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  927 | Tensor<[196, 2048]> self = ?,<br>List[int] size = [1, 14, 14, 2048]            | Done     | N/A                 | N/A          | N/A               | N/A                |
|  928 | Tensor<[196, 3072]> self = ?,<br>List[int] size = [1, 196, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  929 | Tensor<[196, 384]> self = ?,<br>List[int] size = [1, 14, 14, 384]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  930 | Tensor<[196, 384]> self = ?,<br>List[int] size = [4, 49, 384]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  931 | Tensor<[196, 512]> self = ?,<br>List[int] size = [1, 14, 14, 512]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  932 | Tensor<[196, 512]> self = ?,<br>List[int] size = [4, 49, 512]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  933 | Tensor<[196, 768]> self = ?,<br>List[int] size = [1, 196, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  934 | Tensor<[197, 1, 1024]> self = ?,<br>List[int] size = [197, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  935 | Tensor<[197, 1, 1024]> self = ?,<br>List[int] size = [197, 16, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  936 | Tensor<[197, 1, 12, 64]> self = ?,<br>List[int] size = [197, 768]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  937 | Tensor<[197, 1, 16, 64]> self = ?,<br>List[int] size = [197, 1024]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  938 | Tensor<[197, 1, 2304]> self = ?,<br>List[int] size = [197, 1, 3, 768]          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  939 | Tensor<[197, 1, 3072]> self = ?,<br>List[int] size = [197, 1, 3, 1024]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  940 | Tensor<[197, 1, 768]> self = ?,<br>List[int] size = [197, 12, 64]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  941 | Tensor<[197, 1, 768]> self = ?,<br>List[int] size = [197, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  942 | Tensor<[197, 1024]> self = ?,<br>List[int] size = [1, 197, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  943 | Tensor<[197, 1024]> self = ?,<br>List[int] size = [197, 1, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  944 | Tensor<[197, 197, 12]> self = ?,<br>List[int] size = [38809, 12]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  945 | Tensor<[197, 197, 16]> self = ?,<br>List[int] size = [38809, 16]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  946 | Tensor<[197, 197]> self = ?,<br>List[int] size = [-1]                          | Done     | N/A                 | N/A          | N/A               | N/A                |
|  947 | Tensor<[197, 2304]> self = ?,<br>List[int] size = [197, 1, 2304]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  948 | Tensor<[197, 3072]> self = ?,<br>List[int] size = [1, 197, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  949 | Tensor<[197, 3072]> self = ?,<br>List[int] size = [197, 1, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  950 | Tensor<[197, 4096]> self = ?,<br>List[int] size = [1, 197, 4096]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  951 | Tensor<[197, 768]> self = ?,<br>List[int] size = [1, 197, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  952 | Tensor<[197, 768]> self = ?,<br>List[int] size = [197, 1, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  953 | Tensor<[19]> self = ?,<br>List[int] size = [-1, 1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  954 | Tensor<[19]> self = ?,<br>List[int] size = [1, -1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  955 | Tensor<[1]> self = ?,<br>List[int] size = [-1, 1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  956 | Tensor<[1]> self = ?,<br>List[int] size = [1, -1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  957 | Tensor<[1]> self = ?,<br>List[int] size = [1, 1, 1, 1]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  958 | Tensor<[2, 256, 32]> self = ?,<br>List[int] size = [1, 2, 256, 32]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  959 | Tensor<[2, 32, 256]> self = ?,<br>List[int] size = [1, 2, 32, 256]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  960 | Tensor<[2, 4096, 256]> self = ?,<br>List[int] size = [1, 2, 4096, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  961 | Tensor<[2, 4096, 32]> self = ?,<br>List[int] size = [1, 2, 4096, 32]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  962 | Tensor<[2, 4800, 300]> self = ?,<br>List[int] size = [1, 2, 4800, 300]         | Done     | N/A                 | N/A          | N/A               | N/A                |
|  963 | Tensor<[2, 4800, 64]> self = ?,<br>List[int] size = [1, 2, 4800, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
|  964 | Tensor<[2, 7, 2048]> self = ?,<br>List[int] size = [14, 2048]                  | None     | N/A                 | N/A          | N/A               | N/A                |
|  965 | Tensor<[2, 7, 512]> self = ?,<br>List[int] size = [14, 512]                    | None     | N/A                 | N/A          | N/A               | N/A                |
|  966 | Tensor<[2, 7, 512]> self = ?,<br>List[int] size = [2, -1, 8, 64]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  967 | Tensor<[2, 7, 512]> self = ?,<br>List[int] size = [2, 7, 8, 64]                | None     | N/A                 | N/A          | N/A               | N/A                |
|  968 | Tensor<[2, 7, 8, 64]> self = ?,<br>List[int] size = [2, 7, 512]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  969 | Tensor<[2, 7]> self = ?,<br>List[int] size = [-1, 7]                           | None     | N/A                 | N/A          | N/A               | N/A                |
|  970 | Tensor<[2, 8, 7, 64]> self = ?,<br>List[int] size = [16, -1, 64]               | None     | N/A                 | N/A          | N/A               | N/A                |
|  971 | Tensor<[2, 8, 7, 7]> self = ?,<br>List[int] size = [16, 7, 7]                  | None     | N/A                 | N/A          | N/A               | N/A                |
|  972 | Tensor<[201, 3072]> self = ?,<br>List[int] size = [1, 201, 3072]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  973 | Tensor<[201, 768]> self = ?,<br>List[int] size = [1, 201, 768]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  974 | Tensor<[2048, 1280]> self = ?,<br>List[int] size = [1, 2048, 1280]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  975 | Tensor<[2048, 256]> self = ?,<br>List[int] size = [1, 2048, 256]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  976 | Tensor<[2048, 262]> self = ?,<br>List[int] size = [1, 2048, 262]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  977 | Tensor<[2048, 768]> self = ?,<br>List[int] size = [1, 2048, 768]               | Done     | N/A                 | N/A          | N/A               | N/A                |
|  978 | Tensor<[2048]> self = ?,<br>List[int] size = [1, -1, 1, 1]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
|  979 | Tensor<[20]> self = ?,<br>List[int] size = [-1, 1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  980 | Tensor<[20]> self = ?,<br>List[int] size = [1, -1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  981 | Tensor<[221, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                      | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  982 | Tensor<[221, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                      | Unknown  | N/A                 | N/A          | N/A               | N/A                |
|  983 | Tensor<[225, 12]> self = ?,<br>List[int] size = [1, 15, 15, 12]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  984 | Tensor<[225, 16]> self = ?,<br>List[int] size = [1, 15, 15, 16]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  985 | Tensor<[225, 24]> self = ?,<br>List[int] size = [1, 15, 15, 24]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  986 | Tensor<[225, 32]> self = ?,<br>List[int] size = [1, 15, 15, 32]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  987 | Tensor<[225, 3]> self = ?,<br>List[int] size = [1, 15, 15, 3]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  988 | Tensor<[225, 4]> self = ?,<br>List[int] size = [1, 15, 15, 4]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  989 | Tensor<[225, 512]> self = ?,<br>List[int] size = [1, 15, 15, 512]              | Done     | N/A                 | N/A          | N/A               | N/A                |
|  990 | Tensor<[225, 6]> self = ?,<br>List[int] size = [1, 15, 15, 6]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  991 | Tensor<[225, 8]> self = ?,<br>List[int] size = [1, 15, 15, 8]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
|  992 | Tensor<[24, 12, 24]> self = ?,<br>List[int] size = [24, 12, 24]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  993 | Tensor<[24, 12, 64]> self = ?,<br>List[int] size = [24, 12, 64]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  994 | Tensor<[24, 3072]> self = ?,<br>List[int] size = [1, 24, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
|  995 | Tensor<[24, 49, 32]> self = ?,<br>List[int] size = [1, 24, 49, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  996 | Tensor<[24, 49, 49]> self = ?,<br>List[int] size = [1, 24, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  997 | Tensor<[24, 64, 24]> self = ?,<br>List[int] size = [24, 64, 24]                | Done     | N/A                 | N/A          | N/A               | N/A                |
|  998 | Tensor<[24, 64, 32]> self = ?,<br>List[int] size = [1, 24, 64, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
|  999 | Tensor<[24, 64, 64]> self = ?,<br>List[int] size = [1, 24, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1000 | Tensor<[24, 768]> self = ?,<br>List[int] size = [1, 24, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1001 | Tensor<[2401, 12]> self = ?,<br>List[int] size = [49, 49, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1002 | Tensor<[2401, 16]> self = ?,<br>List[int] size = [49, 49, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1003 | Tensor<[2401, 24]> self = ?,<br>List[int] size = [49, 49, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1004 | Tensor<[2401, 32]> self = ?,<br>List[int] size = [49, 49, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1005 | Tensor<[2401, 3]> self = ?,<br>List[int] size = [49, 49, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1006 | Tensor<[2401, 4]> self = ?,<br>List[int] size = [49, 49, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1007 | Tensor<[2401, 6]> self = ?,<br>List[int] size = [49, 49, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1008 | Tensor<[2401, 8]> self = ?,<br>List[int] size = [49, 49, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1009 | Tensor<[24576, 1]> self = ?,<br>List[int] size = [-1]                          | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1010 | Tensor<[25, 12]> self = ?,<br>List[int] size = [-1, 2]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1011 | Tensor<[25, 2]> self = ?,<br>List[int] size = [1, 25, 2]                       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1012 | Tensor<[25, 3072]> self = ?,<br>List[int] size = [1, 25, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1013 | Tensor<[25, 768]> self = ?,<br>List[int] size = [1, 25, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1014 | Tensor<[256, 10240]> self = ?,<br>List[int] size = [1, 256, 10240]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1015 | Tensor<[256, 1024]> self = ?,<br>List[int] size = [1, 256, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1016 | Tensor<[256, 1152]> self = ?,<br>List[int] size = [4, 64, 1152]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1017 | Tensor<[256, 1280]> self = ?,<br>List[int] size = [1, 256, 1280]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1018 | Tensor<[256, 1536]> self = ?,<br>List[int] size = [1, 16, 16, 1536]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1019 | Tensor<[256, 1536]> self = ?,<br>List[int] size = [4, 64, 1536]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1020 | Tensor<[256, 160]> self = ?,<br>List[int] size = [1, 256, 160]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1021 | Tensor<[256, 2048]> self = ?,<br>List[int] size = [1, 16, 16, 2048]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1022 | Tensor<[256, 256]> self = ?,<br>List[int] size = [1, 256, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1023 | Tensor<[256, 2]> self = ?,<br>List[int] size = [1, 256, 2]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1024 | Tensor<[256, 32]> self = ?,<br>List[int] size = [1, 256, 32]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1025 | Tensor<[256, 384]> self = ?,<br>List[int] size = [1, 16, 16, 384]              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1026 | Tensor<[256, 384]> self = ?,<br>List[int] size = [4, 64, 384]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1027 | Tensor<[256, 4096]> self = ?,<br>List[int] size = [1, 256, 4096]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1028 | Tensor<[256, 49, 32]> self = ?,<br>List[int] size = [64, 4, 49, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1029 | Tensor<[256, 49, 49]> self = ?,<br>List[int] size = [64, 4, 49, 49]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1030 | Tensor<[256, 512]> self = ?,<br>List[int] size = [1, 16, 16, 512]              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1031 | Tensor<[256, 512]> self = ?,<br>List[int] size = [1, 256, 512]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1032 | Tensor<[256, 512]> self = ?,<br>List[int] size = [4, 64, 512]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1033 | Tensor<[256, 64, 32]> self = ?,<br>List[int] size = [64, 4, 64, 32]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1034 | Tensor<[256, 64, 64]> self = ?,<br>List[int] size = [64, 4, 64, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1035 | Tensor<[256, 64]> self = ?,<br>List[int] size = [1, 256, 64]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1036 | Tensor<[256, 768]> self = ?,<br>List[int] size = [1, 256, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1037 | Tensor<[256]> self = ?,<br>List[int] size = [1, -1, 1, 1]                      | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1038 | Tensor<[25]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1039 | Tensor<[28, 28]> self = ?,<br>List[int] size = [4, 7, 4, 7]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1040 | Tensor<[2]> self = ?,<br>List[int] size = [-1, 1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1041 | Tensor<[2]> self = ?,<br>List[int] size = [1, -1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1042 | Tensor<[3, 1445, 1445]> self = ?,<br>List[int] size = [1, 3, 1445, 1445]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1043 | Tensor<[3, 1445, 64]> self = ?,<br>List[int] size = [1, 3, 1445, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1044 | Tensor<[300, 128]> self = ?,<br>List[int] size = [1, 300, 128]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1045 | Tensor<[300, 2048]> self = ?,<br>List[int] size = [1, 300, 2048]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1046 | Tensor<[300, 320]> self = ?,<br>List[int] size = [1, 300, 320]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1047 | Tensor<[300, 512]> self = ?,<br>List[int] size = [1, 300, 512]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1048 | Tensor<[300, 64]> self = ?,<br>List[int] size = [1, 300, 64]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1049 | Tensor<[3136, 128]> self = ?,<br>List[int] size = [1, 56, 56, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1050 | Tensor<[3136, 128]> self = ?,<br>List[int] size = [64, 49, 128]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1051 | Tensor<[3136, 288]> self = ?,<br>List[int] size = [64, 49, 288]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1052 | Tensor<[3136, 384]> self = ?,<br>List[int] size = [1, 56, 56, 384]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1053 | Tensor<[3136, 384]> self = ?,<br>List[int] size = [64, 49, 384]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1054 | Tensor<[3136, 512]> self = ?,<br>List[int] size = [1, 56, 56, 512]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1055 | Tensor<[3136, 96]> self = ?,<br>List[int] size = [1, 56, 56, 96]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1056 | Tensor<[3136, 96]> self = ?,<br>List[int] size = [64, 49, 96]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1057 | Tensor<[32, 1536]> self = ?,<br>List[int] size = [1, 32, 1536]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1058 | Tensor<[32, 250880]> self = ?,<br>List[int] size = [1, 32, 250880]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1059 | Tensor<[32, 32]> self = ?,<br>List[int] size = [4, 8, 4, 8]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1060 | Tensor<[32, 4608]> self = ?,<br>List[int] size = [1, 32, 4608]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1061 | Tensor<[32, 49, 32]> self = ?,<br>List[int] size = [1, 32, 49, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1062 | Tensor<[32, 49, 49]> self = ?,<br>List[int] size = [1, 32, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1063 | Tensor<[32, 6144]> self = ?,<br>List[int] size = [1, 32, 6144]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1064 | Tensor<[32, 64, 32]> self = ?,<br>List[int] size = [1, 32, 64, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1065 | Tensor<[32, 64, 64]> self = ?,<br>List[int] size = [1, 32, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1066 | Tensor<[3234, 1, 4]> self = ?,<br>List[int] size = [3234, 4]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1067 | Tensor<[3234, 2, 2]> self = ?,<br>List[int] size = [3234, 4]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1068 | Tensor<[32]> self = ?,<br>List[int] size = [1, 1, 32, 1]                       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1069 | Tensor<[3400, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1070 | Tensor<[3400, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1071 | Tensor<[34]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1072 | Tensor<[361, 12]> self = ?,<br>List[int] size = [-1, 2]                        | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1073 | Tensor<[38809, 12]> self = ?,<br>List[int] size = [197, 197, -1]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1074 | Tensor<[38809, 16]> self = ?,<br>List[int] size = [197, 197, -1]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1075 | Tensor<[38]> self = ?,<br>List[int] size = [-1, 1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1076 | Tensor<[38]> self = ?,<br>List[int] size = [1, -1]                             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1077 | Tensor<[3]> self = ?,<br>List[int] size = [-1, 1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1078 | Tensor<[3]> self = ?,<br>List[int] size = [1, -1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1079 | Tensor<[4, 12, 49, 49]> self = ?,<br>List[int] size = [1, 4, 12, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1080 | Tensor<[4, 12, 49, 49]> self = ?,<br>List[int] size = [48, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1081 | Tensor<[4, 12, 64, 64]> self = ?,<br>List[int] size = [1, 4, 12, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1082 | Tensor<[4, 12, 64, 64]> self = ?,<br>List[int] size = [48, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1083 | Tensor<[4, 12]> self = ?,<br>List[int] size = [-1, 2]                          | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1084 | Tensor<[4, 16, 49, 49]> self = ?,<br>List[int] size = [1, 4, 16, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1085 | Tensor<[4, 16, 49, 49]> self = ?,<br>List[int] size = [64, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1086 | Tensor<[4, 16, 64, 64]> self = ?,<br>List[int] size = [1, 4, 16, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1087 | Tensor<[4, 16, 64, 64]> self = ?,<br>List[int] size = [64, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1088 | Tensor<[4, 3072]> self = ?,<br>List[int] size = [1, 4, 3072]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1089 | Tensor<[4, 49, 1152]> self = ?,<br>List[int] size = [4, 49, 3, 12, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1090 | Tensor<[4, 49, 1536]> self = ?,<br>List[int] size = [4, 49, 3, 16, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1091 | Tensor<[4, 49, 384]> self = ?,<br>List[int] size = [1, 2, 2, 7, 7, 384]        | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1092 | Tensor<[4, 49, 384]> self = ?,<br>List[int] size = [196, 384]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1093 | Tensor<[4, 49, 512]> self = ?,<br>List[int] size = [1, 2, 2, 7, 7, 512]        | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1094 | Tensor<[4, 49, 512]> self = ?,<br>List[int] size = [196, 512]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1095 | Tensor<[4, 51865]> self = ?,<br>List[int] size = [1, 4, 51865]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1096 | Tensor<[4, 64, 1152]> self = ?,<br>List[int] size = [4, 64, 3, 12, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1097 | Tensor<[4, 64, 1536]> self = ?,<br>List[int] size = [4, 64, 3, 16, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1098 | Tensor<[4, 64, 384]> self = ?,<br>List[int] size = [1, 2, 2, 8, 8, 384]        | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1099 | Tensor<[4, 64, 384]> self = ?,<br>List[int] size = [256, 384]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1100 | Tensor<[4, 64, 512]> self = ?,<br>List[int] size = [1, 2, 2, 8, 8, 512]        | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1101 | Tensor<[4, 64, 512]> self = ?,<br>List[int] size = [256, 512]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1102 | Tensor<[4, 768]> self = ?,<br>List[int] size = [1, 4, 768]                     | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1103 | Tensor<[400, 12]> self = ?,<br>List[int] size = [-1, 2]                        | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1104 | Tensor<[4096, 128]> self = ?,<br>List[int] size = [1, 64, 64, 128]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1105 | Tensor<[4096, 128]> self = ?,<br>List[int] size = [64, 64, 128]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1106 | Tensor<[4096, 12]> self = ?,<br>List[int] size = [64, 64, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1107 | Tensor<[4096, 16]> self = ?,<br>List[int] size = [64, 64, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1108 | Tensor<[4096, 24]> self = ?,<br>List[int] size = [64, 64, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1109 | Tensor<[4096, 2560]> self = ?,<br>List[int] size = [1, 4096, 2560]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1110 | Tensor<[4096, 256]> self = ?,<br>List[int] size = [1, 4096, 256]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1111 | Tensor<[4096, 288]> self = ?,<br>List[int] size = [64, 64, 288]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1112 | Tensor<[4096, 320]> self = ?,<br>List[int] size = [1, 4096, 320]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1113 | Tensor<[4096, 32]> self = ?,<br>List[int] size = [64, 64, -1]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1114 | Tensor<[4096, 384]> self = ?,<br>List[int] size = [1, 64, 64, 384]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1115 | Tensor<[4096, 384]> self = ?,<br>List[int] size = [64, 64, 384]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1116 | Tensor<[4096, 3]> self = ?,<br>List[int] size = [64, 64, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1117 | Tensor<[4096, 4]> self = ?,<br>List[int] size = [64, 64, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1118 | Tensor<[4096, 512]> self = ?,<br>List[int] size = [1, 64, 64, 512]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1119 | Tensor<[4096, 64]> self = ?,<br>List[int] size = [1, 4096, 64]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1120 | Tensor<[4096, 6]> self = ?,<br>List[int] size = [64, 64, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1121 | Tensor<[4096, 8]> self = ?,<br>List[int] size = [64, 64, -1]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1122 | Tensor<[4096, 96]> self = ?,<br>List[int] size = [1, 64, 64, 96]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1123 | Tensor<[4096, 96]> self = ?,<br>List[int] size = [64, 64, 96]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1124 | Tensor<[42]> self = ?,<br>List[int] size = [1, 1, 1, 42]                       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1125 | Tensor<[45, 3072]> self = ?,<br>List[int] size = [1, 45, 3072]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1126 | Tensor<[45, 50257]> self = ?,<br>List[int] size = [1, 45, 50257]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1127 | Tensor<[45, 768]> self = ?,<br>List[int] size = [1, 45, 768]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1128 | Tensor<[48, 49, 32]> self = ?,<br>List[int] size = [4, 12, 49, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1129 | Tensor<[48, 49, 49]> self = ?,<br>List[int] size = [4, 12, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1130 | Tensor<[48, 64, 32]> self = ?,<br>List[int] size = [4, 12, 64, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1131 | Tensor<[48, 64, 64]> self = ?,<br>List[int] size = [4, 12, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1132 | Tensor<[4800, 128]> self = ?,<br>List[int] size = [1, 4800, 128]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1133 | Tensor<[4800, 512]> self = ?,<br>List[int] size = [1, 4800, 512]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1134 | Tensor<[49, 1024]> self = ?,<br>List[int] size = [1, 49, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1135 | Tensor<[49, 1024]> self = ?,<br>List[int] size = [1, 7, 7, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1136 | Tensor<[49, 2304]> self = ?,<br>List[int] size = [1, 49, 2304]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1137 | Tensor<[49, 3072]> self = ?,<br>List[int] size = [1, 49, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1138 | Tensor<[49, 3072]> self = ?,<br>List[int] size = [1, 7, 7, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1139 | Tensor<[49, 4096]> self = ?,<br>List[int] size = [1, 7, 7, 4096]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1140 | Tensor<[49, 768]> self = ?,<br>List[int] size = [1, 49, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1141 | Tensor<[49, 768]> self = ?,<br>List[int] size = [1, 7, 7, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1142 | Tensor<[5, 1024, 256]> self = ?,<br>List[int] size = [1, 5, 1024, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1143 | Tensor<[5, 1024, 32]> self = ?,<br>List[int] size = [1, 5, 1024, 32]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1144 | Tensor<[5, 1024]> self = ?,<br>List[int] size = [1, 5, 1024]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1145 | Tensor<[5, 1200, 300]> self = ?,<br>List[int] size = [1, 5, 1200, 300]         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1146 | Tensor<[5, 1200, 64]> self = ?,<br>List[int] size = [1, 5, 1200, 64]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1147 | Tensor<[5, 256, 32]> self = ?,<br>List[int] size = [1, 5, 256, 32]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1148 | Tensor<[5, 3072]> self = ?,<br>List[int] size = [1, 5, 3072]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1149 | Tensor<[5, 32, 256]> self = ?,<br>List[int] size = [1, 5, 32, 256]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1150 | Tensor<[5, 4096]> self = ?,<br>List[int] size = [1, 5, 4096]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1151 | Tensor<[5, 51200]> self = ?,<br>List[int] size = [1, 5, 51200]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1152 | Tensor<[50, 1, 1024]> self = ?,<br>List[int] size = [50, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1153 | Tensor<[50, 1, 1024]> self = ?,<br>List[int] size = [50, 16, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1154 | Tensor<[50, 1, 12, 64]> self = ?,<br>List[int] size = [50, 768]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1155 | Tensor<[50, 1, 16, 64]> self = ?,<br>List[int] size = [50, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1156 | Tensor<[50, 1, 2304]> self = ?,<br>List[int] size = [50, 1, 3, 768]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1157 | Tensor<[50, 1, 3072]> self = ?,<br>List[int] size = [50, 1, 3, 1024]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1158 | Tensor<[50, 1, 768]> self = ?,<br>List[int] size = [50, 12, 64]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1159 | Tensor<[50, 1, 768]> self = ?,<br>List[int] size = [50, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1160 | Tensor<[50, 1024]> self = ?,<br>List[int] size = [1, 50, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1161 | Tensor<[50, 1024]> self = ?,<br>List[int] size = [50, 1, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1162 | Tensor<[50, 2304]> self = ?,<br>List[int] size = [50, 1, 2304]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1163 | Tensor<[50, 3072]> self = ?,<br>List[int] size = [1, 50, 3072]                 | None     | N/A                 | N/A          | N/A               | N/A                |
| 1164 | Tensor<[50, 3072]> self = ?,<br>List[int] size = [50, 1, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1165 | Tensor<[50, 4096]> self = ?,<br>List[int] size = [1, 50, 4096]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1166 | Tensor<[50, 768]> self = ?,<br>List[int] size = [1, 50, 768]                   | None     | N/A                 | N/A          | N/A               | N/A                |
| 1167 | Tensor<[50, 768]> self = ?,<br>List[int] size = [50, 1, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1168 | Tensor<[50]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1169 | Tensor<[512]> self = ?,<br>List[int] size = [1, -1, 1, 1]                      | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1170 | Tensor<[56, 56]> self = ?,<br>List[int] size = [8, 7, 8, 7]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1171 | Tensor<[59, 1024]> self = ?,<br>List[int] size = [1, 59, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1172 | Tensor<[59, 50272]> self = ?,<br>List[int] size = [1, 59, 50272]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1173 | Tensor<[59, 512]> self = ?,<br>List[int] size = [1, 59, 512]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1174 | Tensor<[5]> self = ?,<br>List[int] size = [-1, 1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1175 | Tensor<[5]> self = ?,<br>List[int] size = [1, -1]                              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1176 | Tensor<[6, 1, 100, 256]> self = ?,<br>List[int] size = [600, 256]              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1177 | Tensor<[6, 1, 15]> self = ?,<br>List[int] size = [1, 6, 1, 15]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1178 | Tensor<[6, 1, 17]> self = ?,<br>List[int] size = [1, 6, 1, 17]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1179 | Tensor<[6, 1, 1]> self = ?,<br>List[int] size = [1, 6, 1, 1]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1180 | Tensor<[6, 1, 2]> self = ?,<br>List[int] size = [1, 6, 1, 2]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1181 | Tensor<[6, 1, 64]> self = ?,<br>List[int] size = [1, 6, 1, 64]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1182 | Tensor<[6, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 6, 1, <s0 + 1>]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1183 | Tensor<[6, 15, 15]> self = ?,<br>List[int] size = [1, 6, 15, 15]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1184 | Tensor<[6, 15, 64]> self = ?,<br>List[int] size = [1, 6, 15, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1185 | Tensor<[600, 256]> self = ?,<br>List[int] size = [6, 1, 100, 256]              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1186 | Tensor<[600, 4]> self = ?,<br>List[int] size = [6, 1, 100, 4]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1187 | Tensor<[600, 92]> self = ?,<br>List[int] size = [6, 1, 100, 92]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1188 | Tensor<[63, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1189 | Tensor<[63, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1190 | Tensor<[64, 10240]> self = ?,<br>List[int] size = [1, 64, 10240]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1191 | Tensor<[64, 1024]> self = ?,<br>List[int] size = [1, 64, 1024]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1192 | Tensor<[64, 1024]> self = ?,<br>List[int] size = [1, 8, 8, 1024]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1193 | Tensor<[64, 1280]> self = ?,<br>List[int] size = [1, 64, 1280]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1194 | Tensor<[64, 2304]> self = ?,<br>List[int] size = [1, 64, 2304]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1195 | Tensor<[64, 3, 49, 49]> self = ?,<br>List[int] size = [1, 64, 3, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1196 | Tensor<[64, 3, 49, 49]> self = ?,<br>List[int] size = [192, 49, 49]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1197 | Tensor<[64, 3, 64, 64]> self = ?,<br>List[int] size = [1, 64, 3, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1198 | Tensor<[64, 3, 64, 64]> self = ?,<br>List[int] size = [192, 64, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1199 | Tensor<[64, 3072]> self = ?,<br>List[int] size = [1, 64, 3072]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1200 | Tensor<[64, 3072]> self = ?,<br>List[int] size = [1, 8, 8, 3072]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1201 | Tensor<[64, 4, 49, 49]> self = ?,<br>List[int] size = [1, 64, 4, 49, 49]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1202 | Tensor<[64, 4, 49, 49]> self = ?,<br>List[int] size = [256, 49, 49]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1203 | Tensor<[64, 4, 64, 64]> self = ?,<br>List[int] size = [1, 64, 4, 64, 64]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1204 | Tensor<[64, 4, 64, 64]> self = ?,<br>List[int] size = [256, 64, 64]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1205 | Tensor<[64, 4096]> self = ?,<br>List[int] size = [1, 8, 8, 4096]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1206 | Tensor<[64, 49, 128]> self = ?,<br>List[int] size = [1, 8, 8, 7, 7, 128]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1207 | Tensor<[64, 49, 128]> self = ?,<br>List[int] size = [3136, 128]                | None     | N/A                 | N/A          | N/A               | N/A                |
| 1208 | Tensor<[64, 49, 288]> self = ?,<br>List[int] size = [64, 49, 3, 3, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1209 | Tensor<[64, 49, 32]> self = ?,<br>List[int] size = [4, 16, 49, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1210 | Tensor<[64, 49, 384]> self = ?,<br>List[int] size = [64, 49, 3, 4, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1211 | Tensor<[64, 49, 49]> self = ?,<br>List[int] size = [4, 16, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1212 | Tensor<[64, 49, 96]> self = ?,<br>List[int] size = [1, 8, 8, 7, 7, 96]         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1213 | Tensor<[64, 49, 96]> self = ?,<br>List[int] size = [3136, 96]                  | None     | N/A                 | N/A          | N/A               | N/A                |
| 1214 | Tensor<[64, 64, 128]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 128]       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1215 | Tensor<[64, 64, 128]> self = ?,<br>List[int] size = [4096, 128]                | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1216 | Tensor<[64, 64, 288]> self = ?,<br>List[int] size = [64, 64, 3, 3, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1217 | Tensor<[64, 64, 32]> self = ?,<br>List[int] size = [4, 16, 64, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1218 | Tensor<[64, 64, 384]> self = ?,<br>List[int] size = [64, 64, 3, 4, 32]         | None     | N/A                 | N/A          | N/A               | N/A                |
| 1219 | Tensor<[64, 64, 64]> self = ?,<br>List[int] size = [4, 16, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1220 | Tensor<[64, 64, 96]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 96]         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1221 | Tensor<[64, 64, 96]> self = ?,<br>List[int] size = [4096, 96]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1222 | Tensor<[64, 64]> self = ?,<br>List[int] size = [8, 8, 8, 8]                    | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1223 | Tensor<[64, 768]> self = ?,<br>List[int] size = [1, 64, 768]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1224 | Tensor<[64, 768]> self = ?,<br>List[int] size = [1, 8, 8, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1225 | Tensor<[64, 9, 64]> self = ?,<br>List[int] size = [1, 64, 9, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1226 | Tensor<[64, 9, 9]> self = ?,<br>List[int] size = [1, 64, 9, 9]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1227 | Tensor<[64]> self = ?,<br>List[int] size = [1, -1, 1, 1]                       | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1228 | Tensor<[68]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1229 | Tensor<[7, 18176]> self = ?,<br>List[int] size = [1, 7, 18176]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1230 | Tensor<[7, 2304]> self = ?,<br>List[int] size = [1, 7, 2304]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1231 | Tensor<[7, 2]> self = ?,<br>List[int] size = [1, 7, 2]                         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1232 | Tensor<[7, 3072]> self = ?,<br>List[int] size = [1, 7, 3072]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1233 | Tensor<[7, 4544]> self = ?,<br>List[int] size = [1, 7, 4544]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1234 | Tensor<[7, 4672]> self = ?,<br>List[int] size = [1, 7, 4672]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1235 | Tensor<[7, 65024]> self = ?,<br>List[int] size = [1, 7, 65024]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1236 | Tensor<[7, 768]> self = ?,<br>List[int] size = [1, 7, 768]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1237 | Tensor<[71, 7, 64]> self = ?,<br>List[int] size = [1, 71, 7, 64]               | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1238 | Tensor<[71, 7, 7]> self = ?,<br>List[int] size = [1, 71, 7, 7]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1239 | Tensor<[768, 196]> self = ?,<br>List[int] size = [1, 768, 196]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1240 | Tensor<[768, 384]> self = ?,<br>List[int] size = [1, 768, 384]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1241 | Tensor<[784, 1024]> self = ?,<br>List[int] size = [1, 28, 28, 1024]            | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1242 | Tensor<[784, 192]> self = ?,<br>List[int] size = [1, 28, 28, 192]              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1243 | Tensor<[784, 192]> self = ?,<br>List[int] size = [16, 49, 192]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1244 | Tensor<[784, 256]> self = ?,<br>List[int] size = [1, 28, 28, 256]              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1245 | Tensor<[784, 256]> self = ?,<br>List[int] size = [16, 49, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1246 | Tensor<[784, 576]> self = ?,<br>List[int] size = [16, 49, 576]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1247 | Tensor<[784, 768]> self = ?,<br>List[int] size = [1, 28, 28, 768]              | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1248 | Tensor<[784, 768]> self = ?,<br>List[int] size = [16, 49, 768]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1249 | Tensor<[7]> self = ?,<br>List[int] size = [-1, 1]                              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1250 | Tensor<[8, 1, 10]> self = ?,<br>List[int] size = [1, 8, 1, 10]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1251 | Tensor<[8, 1, 1]> self = ?,<br>List[int] size = [1, 8, 1, 1]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1252 | Tensor<[8, 1, 2]> self = ?,<br>List[int] size = [1, 8, 1, 2]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1253 | Tensor<[8, 1, 64]> self = ?,<br>List[int] size = [1, 8, 1, 64]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1254 | Tensor<[8, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 8, 1, <s0 + 1>]       | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1255 | Tensor<[8, 10, 10]> self = ?,<br>List[int] size = [1, 8, 10, 10]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1256 | Tensor<[8, 10, 64]> self = ?,<br>List[int] size = [1, 8, 10, 64]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1257 | Tensor<[8, 2048, 256]> self = ?,<br>List[int] size = [1, 8, 2048, 256]         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1258 | Tensor<[8, 2048, 96]> self = ?,<br>List[int] size = [1, 8, 2048, 96]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1259 | Tensor<[8, 256, 160]> self = ?,<br>List[int] size = [1, 8, 256, 160]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1260 | Tensor<[8, 256, 2048]> self = ?,<br>List[int] size = [1, 8, 256, 2048]         | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1261 | Tensor<[8, 256, 256]> self = ?,<br>List[int] size = [1, 8, 256, 256]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1262 | Tensor<[8, 256, 32]> self = ?,<br>List[int] size = [1, 8, 256, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1263 | Tensor<[8, 300, 300]> self = ?,<br>List[int] size = [1, 8, 300, 300]           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1264 | Tensor<[8, 300, 64]> self = ?,<br>List[int] size = [1, 8, 300, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1265 | Tensor<[8, 32, 256]> self = ?,<br>List[int] size = [1, 8, 32, 256]             | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1266 | Tensor<[850, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                      | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1267 | Tensor<[850, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                      | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1268 | Tensor<[8732, 1, 4]> self = ?,<br>List[int] size = [8732, 4]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1269 | Tensor<[8732, 2, 2]> self = ?,<br>List[int] size = [8732, 4]                   | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1270 | Tensor<[9, 1024]> self = ?,<br>List[int] size = [1, 9, 1024]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1271 | Tensor<[9, 1280]> self = ?,<br>List[int] size = [1, 9, 1280]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1272 | Tensor<[9, 128]> self = ?,<br>List[int] size = [1, 9, 128]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1273 | Tensor<[9, 12]> self = ?,<br>List[int] size = [-1, 2]                          | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1274 | Tensor<[9, 16384]> self = ?,<br>List[int] size = [1, 9, 16384]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1275 | Tensor<[9, 2048]> self = ?,<br>List[int] size = [1, 9, 2048]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1276 | Tensor<[9, 30000]> self = ?,<br>List[int] size = [1, 9, 30000]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1277 | Tensor<[9, 3072]> self = ?,<br>List[int] size = [1, 9, 3072]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1278 | Tensor<[9, 320]> self = ?,<br>List[int] size = [1, 9, 320]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1279 | Tensor<[9, 4096]> self = ?,<br>List[int] size = [1, 9, 4096]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1280 | Tensor<[9, 4]> self = ?,<br>List[int] size = [1, -1, 4]                        | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1281 | Tensor<[9, 640]> self = ?,<br>List[int] size = [1, 9, 640]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1282 | Tensor<[9, 768]> self = ?,<br>List[int] size = [1, 9, 768]                     | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1283 | Tensor<[9, 8192]> self = ?,<br>List[int] size = [1, 9, 8192]                   | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1284 | Tensor<[9, 8]> self = ?,<br>List[int] size = [-1, 2]                           | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1285 | Tensor<[920, 1, 2048]> self = ?,<br>List[int] size = [920, 2048]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1286 | Tensor<[920, 1, 256]> self = ?,<br>List[int] size = [920, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1287 | Tensor<[920, 1, 256]> self = ?,<br>List[int] size = [920, 8, 32]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1288 | Tensor<[920, 2048]> self = ?,<br>List[int] size = [920, 1, 2048]               | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1289 | Tensor<[920, 256]> self = ?,<br>List[int] size = [920, 1, 256]                 | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1290 | Tensor<[920, 8, 32]> self = ?,<br>List[int] size = [920, 256]                  | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1291 | Tensor<[96, 49, 32]> self = ?,<br>List[int] size = [16, 6, 49, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1292 | Tensor<[96, 49, 49]> self = ?,<br>List[int] size = [16, 6, 49, 49]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1293 | Tensor<[96, 64, 32]> self = ?,<br>List[int] size = [16, 6, 64, 32]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1294 | Tensor<[96, 64, 64]> self = ?,<br>List[int] size = [16, 6, 64, 64]             | Done     | N/A                 | N/A          | N/A               | N/A                |
| 1295 | Tensor<[9]> self = ?,<br>List[int] size = [1, -1]                              | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1296 | Tensor<[s0, 256]> self = ?,<br>List[int] size = [1, <s0>, 256]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
| 1297 | Tensor<[s0, 768]> self = ?,<br>List[int] size = [1, <s0>, 768]                 | Unknown  | N/A                 | N/A          | N/A               | N/A                |
