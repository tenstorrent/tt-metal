### aten.view.default
|      | ATen Input Variations                                                          | Status   |
|-----:|:-------------------------------------------------------------------------------|:---------|
|    0 | Tensor<[0, 1, 4]> self = ?,<br>List[int] size = [0, 4]                         | Unknown  |
|    1 | Tensor<[0, 2, 2]> self = ?,<br>List[int] size = [0, 4]                         | Unknown  |
|    2 | Tensor<[1, 1, 1, 16, 2]> self = ?,<br>List[int] size = [1, 1, 1, 32]           | Unknown  |
|    3 | Tensor<[1, 1, 1, 4, 4]> self = ?,<br>List[int] size = [1, -1, 4]               | Fallback |
|    4 | Tensor<[1, 1, 1, 4, 91]> self = ?,<br>List[int] size = [1, -1, 91]             | Fallback |
|    5 | Tensor<[1, 1, 1, 6, 4]> self = ?,<br>List[int] size = [1, -1, 4]               | Fallback |
|    6 | Tensor<[1, 1, 1, 6, 91]> self = ?,<br>List[int] size = [1, -1, 91]             | Fallback |
|    7 | Tensor<[1, 1, 1, 7, 7, 1024]> self = ?,<br>List[int] size = [1, 49, 1024]      | Fallback |
|    8 | Tensor<[1, 1, 1, 7, 7, 768]> self = ?,<br>List[int] size = [1, 49, 768]        | Fallback |
|    9 | Tensor<[1, 1, 1, 8, 8, 1024]> self = ?,<br>List[int] size = [1, 64, 1024]      | Fallback |
|   10 | Tensor<[1, 1, 1, 8, 8, 768]> self = ?,<br>List[int] size = [1, 64, 768]        | Fallback |
|   11 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [-1, 1024]                  | Unknown  |
|   12 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]             | Unknown  |
|   13 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, 1, 1024]                | Unknown  |
|   14 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, 1, 16, 64]              | Unknown  |
|   15 | Tensor<[1, 1, 1024]> self = ?,<br>List[int] size = [1, 1024]                   | Unknown  |
|   16 | Tensor<[1, 1, 12, 16, 2]> self = ?,<br>List[int] size = [1, 192, 2]            | Unknown  |
|   17 | Tensor<[1, 1, 12, 16]> self = ?,<br>List[int] size = [1, 192]                  | Unknown  |
|   18 | Tensor<[1, 1, 12, 64]> self = ?,<br>List[int] size = [1, -1, 768]              | Unknown  |
|   19 | Tensor<[1, 1, 12, 64]> self = ?,<br>List[int] size = [1, 1, 768]               | Fallback |
|   20 | Tensor<[1, 1, 1280]> self = ?,<br>List[int] size = [1, 1280]                   | Unknown  |
|   21 | Tensor<[1, 1, 16, 16, 2]> self = ?,<br>List[int] size = [1, 1, 16, 32]         | Unknown  |
|   22 | Tensor<[1, 1, 16, 64]> self = ?,<br>List[int] size = [1, -1, 1024]             | Unknown  |
|   23 | Tensor<[1, 1, 16, 64]> self = ?,<br>List[int] size = [1, 1, 1024]              | Unknown  |
|   24 | Tensor<[1, 1, 16384, 256]> self = ?,<br>List[int] size = [1, 16384, 256]       | Done     |
|   25 | Tensor<[1, 1, 16384, 32]> self = ?,<br>List[int] size = [1, 16384, 32]         | Done     |
|   26 | Tensor<[1, 1, 19200, 300]> self = ?,<br>List[int] size = [1, 19200, 300]       | Fallback |
|   27 | Tensor<[1, 1, 19200, 64]> self = ?,<br>List[int] size = [1, 19200, 64]         | Done     |
|   28 | Tensor<[1, 1, 2048]> self = ?,<br>List[int] size = [1, 2048]                   | Unknown  |
|   29 | Tensor<[1, 1, 256, 32]> self = ?,<br>List[int] size = [1, 256, 32]             | Done     |
|   30 | Tensor<[1, 1, 256]> self = ?,<br>List[int] size = [1, 256]                     | Unknown  |
|   31 | Tensor<[1, 1, 300, 64]> self = ?,<br>List[int] size = [1, 300, 64]             | Done     |
|   32 | Tensor<[1, 1, 3072]> self = ?,<br>List[int] size = [1, 1, 4, -1]               | Unknown  |
|   33 | Tensor<[1, 1, 3072]> self = ?,<br>List[int] size = [1, 3072]                   | Done     |
|   34 | Tensor<[1, 1, 32, 256]> self = ?,<br>List[int] size = [1, 32, 256]             | Done     |
|   35 | Tensor<[1, 1, 384]> self = ?,<br>List[int] size = [1, -1, 6, 64]               | Unknown  |
|   36 | Tensor<[1, 1, 384]> self = ?,<br>List[int] size = [1, 384]                     | Unknown  |
|   37 | Tensor<[1, 1, 4, 256]> self = ?,<br>List[int] size = [1, 1, 4, 4, 64]          | Unknown  |
|   38 | Tensor<[1, 1, 4096]> self = ?,<br>List[int] size = [1, 4096]                   | Unknown  |
|   39 | Tensor<[1, 1, 45]> self = ?,<br>List[int] size = [1, 45]                       | Unknown  |
|   40 | Tensor<[1, 1, 512]> self = ?,<br>List[int] size = [1, -1, 8, 64]               | Unknown  |
|   41 | Tensor<[1, 1, 512]> self = ?,<br>List[int] size = [1, 512]                     | Unknown  |
|   42 | Tensor<[1, 1, 6, 64]> self = ?,<br>List[int] size = [1, -1, 384]               | Unknown  |
|   43 | Tensor<[1, 1, 64, 300]> self = ?,<br>List[int] size = [1, 64, 300]             | Fallback |
|   44 | Tensor<[1, 1, 7, 1, 7, 1024]> self = ?,<br>List[int] size = [1, 7, 7, 1024]    | Fallback |
|   45 | Tensor<[1, 1, 7, 1, 7, 768]> self = ?,<br>List[int] size = [1, 7, 7, 768]      | Fallback |
|   46 | Tensor<[1, 1, 7, 64]> self = ?,<br>List[int] size = [1, 1, 7, 64]              | Unknown  |
|   47 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [-1, 1, 768]                 | Unknown  |
|   48 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]              | Fallback |
|   49 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [1, 1, 12, 64]               | Fallback |
|   50 | Tensor<[1, 1, 768]> self = ?,<br>List[int] size = [1, 768]                     | Done     |
|   51 | Tensor<[1, 1, 8, 1, 8, 1024]> self = ?,<br>List[int] size = [1, 8, 8, 1024]    | Fallback |
|   52 | Tensor<[1, 1, 8, 1, 8, 768]> self = ?,<br>List[int] size = [1, 8, 8, 768]      | Fallback |
|   53 | Tensor<[1, 1, 8, 64]> self = ?,<br>List[int] size = [1, -1, 512]               | Unknown  |
|   54 | Tensor<[1, 1, 80]> self = ?,<br>List[int] size = [1, 80]                       | Unknown  |
|   55 | Tensor<[1, 10, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]            | Fallback |
|   56 | Tensor<[1, 10, 1024]> self = ?,<br>List[int] size = [10, 1024]                 | Done     |
|   57 | Tensor<[1, 10, 12, 64]> self = ?,<br>List[int] size = [1, -1, 768]             | Fallback |
|   58 | Tensor<[1, 10, 12, 64]> self = ?,<br>List[int] size = [1, 10, 768]             | Fallback |
|   59 | Tensor<[1, 10, 16, 64]> self = ?,<br>List[int] size = [1, -1, 1024]            | Fallback |
|   60 | Tensor<[1, 10, 2048]> self = ?,<br>List[int] size = [10, 2048]                 | Done     |
|   61 | Tensor<[1, 10, 3072]> self = ?,<br>List[int] size = [10, 3072]                 | Done     |
|   62 | Tensor<[1, 10, 4096]> self = ?,<br>List[int] size = [10, 4096]                 | Done     |
|   63 | Tensor<[1, 10, 512]> self = ?,<br>List[int] size = [1, -1, 8, 64]              | Fallback |
|   64 | Tensor<[1, 10, 512]> self = ?,<br>List[int] size = [10, 512]                   | Done     |
|   65 | Tensor<[1, 10, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | Fallback |
|   66 | Tensor<[1, 10, 768]> self = ?,<br>List[int] size = [1, 10, 12, 64]             | Fallback |
|   67 | Tensor<[1, 10, 768]> self = ?,<br>List[int] size = [10, 768]                   | Done     |
|   68 | Tensor<[1, 10, 8, 64]> self = ?,<br>List[int] size = [1, -1, 512]              | Fallback |
|   69 | Tensor<[1, 100, 192]> self = ?,<br>List[int] size = [100, 192]                 | Done     |
|   70 | Tensor<[1, 1000, 1, 1]> self = ?,<br>List[int] size = [1, 1000]                | Fallback |
|   71 | Tensor<[1, 1008, 1, 1]> self = ?,<br>List[int] size = [1, 1008]                | Fallback |
|   72 | Tensor<[1, 1024, 1, 1]> self = ?,<br>List[int] size = [1, -1]                  | Fallback |
|   73 | Tensor<[1, 1024, 1, 1]> self = ?,<br>List[int] size = [1, 1024]                | Fallback |
|   74 | Tensor<[1, 1024, 14, 14]> self = ?,<br>List[int] size = [1, 1024, 196]         | Fallback |
|   75 | Tensor<[1, 1024, 16, 16]> self = ?,<br>List[int] size = [1, 1024, 256]         | Fallback |
|   76 | Tensor<[1, 1024, 160]> self = ?,<br>List[int] size = [1, 1024, 5, 32]          | Fallback |
|   77 | Tensor<[1, 1024, 160]> self = ?,<br>List[int] size = [1, 32, 32, -1]           | Fallback |
|   78 | Tensor<[1, 1024, 160]> self = ?,<br>List[int] size = [1024, 160]               | Done     |
|   79 | Tensor<[1, 1024, 2560]> self = ?,<br>List[int] size = [1024, 2560]             | Done     |
|   80 | Tensor<[1, 1024, 256]> self = ?,<br>List[int] size = [1, 1024, 16, 16]         | Fallback |
|   81 | Tensor<[1, 1024, 5, 32]> self = ?,<br>List[int] size = [1, 1024, 160]          | Fallback |
|   82 | Tensor<[1, 1024, 640]> self = ?,<br>List[int] size = [1, -1, 8, 80]            | Fallback |
|   83 | Tensor<[1, 1024, 640]> self = ?,<br>List[int] size = [1, 32, 32, 640]          | Fallback |
|   84 | Tensor<[1, 1024, 640]> self = ?,<br>List[int] size = [1024, 640]               | Done     |
|   85 | Tensor<[1, 1024, 7, 7]> self = ?,<br>List[int] size = [1, 1024, 49]            | Fallback |
|   86 | Tensor<[1, 1024, 8, 80]> self = ?,<br>List[int] size = [1, -1, 640]            | Fallback |
|   87 | Tensor<[1, 1024]> self = ?,<br>List[int] size = [1, 1, 1024]                   | Unknown  |
|   88 | Tensor<[1, 10]> self = ?,<br>List[int] size = [-1, 10]                         | Fallback |
|   89 | Tensor<[1, 10]> self = ?,<br>List[int] size = [10]                             | Done     |
|   90 | Tensor<[1, 12, 1, 10]> self = ?,<br>List[int] size = [12, 1, 10]               | Unknown  |
|   91 | Tensor<[1, 12, 1, 1]> self = ?,<br>List[int] size = [12, 1, 1]                 | Unknown  |
|   92 | Tensor<[1, 12, 1, 24]> self = ?,<br>List[int] size = [12, 1, 24]               | Unknown  |
|   93 | Tensor<[1, 12, 1, 2]> self = ?,<br>List[int] size = [12, 1, 2]                 | Unknown  |
|   94 | Tensor<[1, 12, 1, 46]> self = ?,<br>List[int] size = [12, 1, 46]               | Unknown  |
|   95 | Tensor<[1, 12, 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]              | Unknown  |
|   96 | Tensor<[1, 12, 1, 64]> self = ?,<br>List[int] size = [12, 1, 64]               | Unknown  |
|   97 | Tensor<[1, 12, 1, s0 + 1]> self = ?,<br>List[int] size = [12, 1, <s0 + 1>]     | Unknown  |
|   98 | Tensor<[1, 12, 1, s10 + 1]> self = ?,<br>List[int] size = [12, 1, <s10 + 1>]   | Unknown  |
|   99 | Tensor<[1, 12, 10, 10]> self = ?,<br>List[int] size = [12, 10, 10]             | Done     |
|  100 | Tensor<[1, 12, 10, 64]> self = ?,<br>List[int] size = [12, 10, 64]             | Done     |
|  101 | Tensor<[1, 12, 12, 12]> self = ?,<br>List[int] size = [12, 12, 12]             | Done     |
|  102 | Tensor<[1, 12, 12, 64]> self = ?,<br>List[int] size = [12, 12, 64]             | Done     |
|  103 | Tensor<[1, 12, 128]> self = ?,<br>List[int] size = [12, 128]                   | Done     |
|  104 | Tensor<[1, 12, 14, 14]> self = ?,<br>List[int] size = [12, 14, 14]             | Done     |
|  105 | Tensor<[1, 12, 14, 64]> self = ?,<br>List[int] size = [12, 14, 64]             | Done     |
|  106 | Tensor<[1, 12, 16, 16]> self = ?,<br>List[int] size = [12, 16, 16]             | Done     |
|  107 | Tensor<[1, 12, 16, 64]> self = ?,<br>List[int] size = [12, 16, 64]             | Done     |
|  108 | Tensor<[1, 12, 197, 197]> self = ?,<br>List[int] size = [12, 197, 197]         | Fallback |
|  109 | Tensor<[1, 12, 197, 64]> self = ?,<br>List[int] size = [12, 197, 64]           | Done     |
|  110 | Tensor<[1, 12, 2, 64]> self = ?,<br>List[int] size = [12, -1, 64]              | Unknown  |
|  111 | Tensor<[1, 12, 2, 64]> self = ?,<br>List[int] size = [12, 2, 64]               | Unknown  |
|  112 | Tensor<[1, 12, 201, 201]> self = ?,<br>List[int] size = [12, 201, 201]         | Unknown  |
|  113 | Tensor<[1, 12, 201, 64]> self = ?,<br>List[int] size = [12, 201, 64]           | Unknown  |
|  114 | Tensor<[1, 12, 24, 24]> self = ?,<br>List[int] size = [12, 24, 24]             | Done     |
|  115 | Tensor<[1, 12, 24, 64]> self = ?,<br>List[int] size = [12, -1, 64]             | Fallback |
|  116 | Tensor<[1, 12, 25, 25]> self = ?,<br>List[int] size = [12, 25, 25]             | Done     |
|  117 | Tensor<[1, 12, 25, 64]> self = ?,<br>List[int] size = [12, 25, 64]             | Done     |
|  118 | Tensor<[1, 12, 3072]> self = ?,<br>List[int] size = [12, 3072]                 | Done     |
|  119 | Tensor<[1, 12, 45, 45]> self = ?,<br>List[int] size = [12, 45, 45]             | Unknown  |
|  120 | Tensor<[1, 12, 45, 64]> self = ?,<br>List[int] size = [12, 45, 64]             | Unknown  |
|  121 | Tensor<[1, 12, 46, 64]> self = ?,<br>List[int] size = [12, 46, 64]             | Unknown  |
|  122 | Tensor<[1, 12, 50, 64]> self = ?,<br>List[int] size = [12, -1, 64]             | Fallback |
|  123 | Tensor<[1, 12, 64, 10]> self = ?,<br>List[int] size = [12, 64, 10]             | Done     |
|  124 | Tensor<[1, 12, 64, 12]> self = ?,<br>List[int] size = [12, 64, 12]             | Done     |
|  125 | Tensor<[1, 12, 64, 14]> self = ?,<br>List[int] size = [12, 64, 14]             | Done     |
|  126 | Tensor<[1, 12, 64, 16]> self = ?,<br>List[int] size = [12, 64, 16]             | Done     |
|  127 | Tensor<[1, 12, 64, 197]> self = ?,<br>List[int] size = [12, 64, 197]           | Fallback |
|  128 | Tensor<[1, 12, 64, 1]> self = ?,<br>List[int] size = [12, 64, 1]               | Unknown  |
|  129 | Tensor<[1, 12, 64, 201]> self = ?,<br>List[int] size = [12, 64, 201]           | Unknown  |
|  130 | Tensor<[1, 12, 64, 25]> self = ?,<br>List[int] size = [12, 64, 25]             | Done     |
|  131 | Tensor<[1, 12, 64, 2]> self = ?,<br>List[int] size = [12, 64, 2]               | Unknown  |
|  132 | Tensor<[1, 12, 64, 45]> self = ?,<br>List[int] size = [12, 64, 45]             | Unknown  |
|  133 | Tensor<[1, 12, 64, 46]> self = ?,<br>List[int] size = [12, 64, 46]             | Unknown  |
|  134 | Tensor<[1, 12, 64, 7]> self = ?,<br>List[int] size = [12, 64, 7]               | Unknown  |
|  135 | Tensor<[1, 12, 64, 9]> self = ?,<br>List[int] size = [12, 64, 9]               | Done     |
|  136 | Tensor<[1, 12, 64, s0 + 1]> self = ?,<br>List[int] size = [12, 64, <s0 + 1>]   | Unknown  |
|  137 | Tensor<[1, 12, 64, s10 + 1]> self = ?,<br>List[int] size = [12, 64, <s10 + 1>] | Unknown  |
|  138 | Tensor<[1, 12, 7, 64]> self = ?,<br>List[int] size = [12, 7, 64]               | Unknown  |
|  139 | Tensor<[1, 12, 7, 7]> self = ?,<br>List[int] size = [12, 7, 7]                 | Unknown  |
|  140 | Tensor<[1, 12, 768]> self = ?,<br>List[int] size = [1, 12, 12, 64]             | Fallback |
|  141 | Tensor<[1, 12, 768]> self = ?,<br>List[int] size = [12, 768]                   | Done     |
|  142 | Tensor<[1, 12, 9, 64]> self = ?,<br>List[int] size = [12, 9, 64]               | Done     |
|  143 | Tensor<[1, 12, 9, 9]> self = ?,<br>List[int] size = [12, 9, 9]                 | Done     |
|  144 | Tensor<[1, 12, s0 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  |
|  145 | Tensor<[1, 12, s0 + 1, 64]> self = ?,<br>List[int] size = [12, <s0 + 1>, 64]   | Unknown  |
|  146 | Tensor<[1, 12, s10 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]        | Unknown  |
|  147 | Tensor<[1, 12, s10 + 1, 64]> self = ?,<br>List[int] size = [12, <s10 + 1>, 64] | Unknown  |
|  148 | Tensor<[1, 12, s2 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  |
|  149 | Tensor<[1, 12, s4 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  |
|  150 | Tensor<[1, 12, s6 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  |
|  151 | Tensor<[1, 12, s8 + 1, 64]> self = ?,<br>List[int] size = [12, -1, 64]         | Unknown  |
|  152 | Tensor<[1, 1200, 1280]> self = ?,<br>List[int] size = [1200, 1280]             | Done     |
|  153 | Tensor<[1, 1200, 320]> self = ?,<br>List[int] size = [1, 1200, 5, 64]          | Fallback |
|  154 | Tensor<[1, 1200, 320]> self = ?,<br>List[int] size = [1, 30, 40, -1]           | Fallback |
|  155 | Tensor<[1, 1200, 320]> self = ?,<br>List[int] size = [1200, 320]               | Done     |
|  156 | Tensor<[1, 1200, 5, 64]> self = ?,<br>List[int] size = [1, 1200, 320]          | Fallback |
|  157 | Tensor<[1, 128, 128, 128]> self = ?,<br>List[int] size = [1, 128, 16384]       | Fallback |
|  158 | Tensor<[1, 128, 15, 20]> self = ?,<br>List[int] size = [1, 128, 300]           | Fallback |
|  159 | Tensor<[1, 128, 16384]> self = ?,<br>List[int] size = [1, 128, 128, 128]       | Fallback |
|  160 | Tensor<[1, 128, 4800]> self = ?,<br>List[int] size = [1, 128, 60, 80]          | Fallback |
|  161 | Tensor<[1, 128, 60, 80]> self = ?,<br>List[int] size = [1, 128, 4800]          | Fallback |
|  162 | Tensor<[1, 1280, 1, 1]> self = ?,<br>List[int] size = [1, 1280]                | Fallback |
|  163 | Tensor<[1, 1280, 1200]> self = ?,<br>List[int] size = [1, 1280, 30, 40]        | Fallback |
|  164 | Tensor<[1, 1280, 30, 40]> self = ?,<br>List[int] size = [1, 1280, 1200]        | Fallback |
|  165 | Tensor<[1, 1280, 37, 37]> self = ?,<br>List[int] size = [1, 1280, 1369]        | Fallback |
|  166 | Tensor<[1, 128]> self = ?,<br>List[int] size = [128]                           | Done     |
|  167 | Tensor<[1, 12]> self = ?,<br>List[int] size = [-1, 2]                          | Fallback |
|  168 | Tensor<[1, 1370, 1280]> self = ?,<br>List[int] size = [1370, 1280]             | Done     |
|  169 | Tensor<[1, 1370, 5120]> self = ?,<br>List[int] size = [1370, 5120]             | Done     |
|  170 | Tensor<[1, 14, 128]> self = ?,<br>List[int] size = [14, 128]                   | Done     |
|  171 | Tensor<[1, 14, 14, 1024]> self = ?,<br>List[int] size = [196, 1024]            | Fallback |
|  172 | Tensor<[1, 14, 14, 1536]> self = ?,<br>List[int] size = [196, 1536]            | Fallback |
|  173 | Tensor<[1, 14, 14, 2048]> self = ?,<br>List[int] size = [196, 2048]            | Fallback |
|  174 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] size = [1, 2, 7, 2, 7, 384]    | Fallback |
|  175 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] size = [196, 384]              | Fallback |
|  176 | Tensor<[1, 14, 14, 512]> self = ?,<br>List[int] size = [1, 2, 7, 2, 7, 512]    | Fallback |
|  177 | Tensor<[1, 14, 14, 512]> self = ?,<br>List[int] size = [196, 512]              | Fallback |
|  178 | Tensor<[1, 14, 14, 768]> self = ?,<br>List[int] size = [196, 768]              | Fallback |
|  179 | Tensor<[1, 14, 3072]> self = ?,<br>List[int] size = [14, 3072]                 | Done     |
|  180 | Tensor<[1, 14, 768]> self = ?,<br>List[int] size = [1, 14, 12, 64]             | Fallback |
|  181 | Tensor<[1, 14, 768]> self = ?,<br>List[int] size = [14, 768]                   | Done     |
|  182 | Tensor<[1, 1445, 192]> self = ?,<br>List[int] size = [1, 1445, 3, 64]          | Fallback |
|  183 | Tensor<[1, 1445, 192]> self = ?,<br>List[int] size = [1445, 192]               | Done     |
|  184 | Tensor<[1, 1445, 3, 64]> self = ?,<br>List[int] size = [1, 1445, 192]          | Fallback |
|  185 | Tensor<[1, 1445, 768]> self = ?,<br>List[int] size = [1445, 768]               | Done     |
|  186 | Tensor<[1, 15, 1024]> self = ?,<br>List[int] size = [15, 1024]                 | Done     |
|  187 | Tensor<[1, 15, 15, 12]> self = ?,<br>List[int] size = [-1, 12]                 | Fallback |
|  188 | Tensor<[1, 15, 15, 16]> self = ?,<br>List[int] size = [-1, 16]                 | Fallback |
|  189 | Tensor<[1, 15, 15, 24]> self = ?,<br>List[int] size = [-1, 24]                 | Fallback |
|  190 | Tensor<[1, 15, 15, 2]> self = ?,<br>List[int] size = [225, 2]                  | Fallback |
|  191 | Tensor<[1, 15, 15, 32]> self = ?,<br>List[int] size = [-1, 32]                 | Fallback |
|  192 | Tensor<[1, 15, 15, 3]> self = ?,<br>List[int] size = [-1, 3]                   | Fallback |
|  193 | Tensor<[1, 15, 15, 4]> self = ?,<br>List[int] size = [-1, 4]                   | Fallback |
|  194 | Tensor<[1, 15, 15, 512]> self = ?,<br>List[int] size = [225, 512]              | Fallback |
|  195 | Tensor<[1, 15, 15, 6]> self = ?,<br>List[int] size = [-1, 6]                   | Fallback |
|  196 | Tensor<[1, 15, 15, 8]> self = ?,<br>List[int] size = [-1, 8]                   | Fallback |
|  197 | Tensor<[1, 15, 384]> self = ?,<br>List[int] size = [1, -1, 6, 64]              | Fallback |
|  198 | Tensor<[1, 15, 384]> self = ?,<br>List[int] size = [15, 384]                   | Done     |
|  199 | Tensor<[1, 15, 512]> self = ?,<br>List[int] size = [15, 512]                   | Done     |
|  200 | Tensor<[1, 15, 6, 64]> self = ?,<br>List[int] size = [1, -1, 384]              | Fallback |
|  201 | Tensor<[1, 1500, 12, 64]> self = ?,<br>List[int] size = [1, 1500, 768]         | Fallback |
|  202 | Tensor<[1, 1500, 3072]> self = ?,<br>List[int] size = [1500, 3072]             | Done     |
|  203 | Tensor<[1, 1500, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]           | Fallback |
|  204 | Tensor<[1, 1500, 768]> self = ?,<br>List[int] size = [1, 1500, 12, 64]         | Fallback |
|  205 | Tensor<[1, 1500, 768]> self = ?,<br>List[int] size = [1500, 768]               | Done     |
|  206 | Tensor<[1, 1512, 1, 1]> self = ?,<br>List[int] size = [1, 1512]                | Fallback |
|  207 | Tensor<[1, 1536, 1, 1]> self = ?,<br>List[int] size = [1, 1536]                | Fallback |
|  208 | Tensor<[1, 15]> self = ?,<br>List[int] size = [-1, 15]                         | Fallback |
|  209 | Tensor<[1, 16, 1, 10]> self = ?,<br>List[int] size = [16, 1, 10]               | Unknown  |
|  210 | Tensor<[1, 16, 1, 1]> self = ?,<br>List[int] size = [1, -1, 4, 1, 1]           | Fallback |
|  211 | Tensor<[1, 16, 1, 1]> self = ?,<br>List[int] size = [16, 1, 1]                 | Unknown  |
|  212 | Tensor<[1, 16, 1, 2]> self = ?,<br>List[int] size = [16, 1, 2]                 | Unknown  |
|  213 | Tensor<[1, 16, 1, 60]> self = ?,<br>List[int] size = [16, 1, 60]               | Unknown  |
|  214 | Tensor<[1, 16, 1, 64]> self = ?,<br>List[int] size = [16, -1, 64]              | Unknown  |
|  215 | Tensor<[1, 16, 1, 64]> self = ?,<br>List[int] size = [16, 1, 64]               | Unknown  |
|  216 | Tensor<[1, 16, 1, 6]> self = ?,<br>List[int] size = [16, 1, 6]                 | Unknown  |
|  217 | Tensor<[1, 16, 1, s0 + 1]> self = ?,<br>List[int] size = [16, 1, <s0 + 1>]     | Unknown  |
|  218 | Tensor<[1, 16, 1, s10 + 1]> self = ?,<br>List[int] size = [16, 1, <s10 + 1>]   | Unknown  |
|  219 | Tensor<[1, 16, 10, 10]> self = ?,<br>List[int] size = [16, 10, 10]             | Done     |
|  220 | Tensor<[1, 16, 10, 64]> self = ?,<br>List[int] size = [16, 10, 64]             | Done     |
|  221 | Tensor<[1, 16, 12, 64]> self = ?,<br>List[int] size = [1, -1, 768]             | Fallback |
|  222 | Tensor<[1, 16, 128, 9]> self = ?,<br>List[int] size = [16, 128, 9]             | Done     |
|  223 | Tensor<[1, 16, 16, 1024]> self = ?,<br>List[int] size = [256, 1024]            | Fallback |
|  224 | Tensor<[1, 16, 16, 1280]> self = ?,<br>List[int] size = [1, 256, 1280]         | Fallback |
|  225 | Tensor<[1, 16, 16, 1536]> self = ?,<br>List[int] size = [256, 1536]            | Fallback |
|  226 | Tensor<[1, 16, 16, 2048]> self = ?,<br>List[int] size = [256, 2048]            | Fallback |
|  227 | Tensor<[1, 16, 16, 384]> self = ?,<br>List[int] size = [1, 2, 8, 2, 8, 384]    | Fallback |
|  228 | Tensor<[1, 16, 16, 384]> self = ?,<br>List[int] size = [256, 384]              | Fallback |
|  229 | Tensor<[1, 16, 16, 512]> self = ?,<br>List[int] size = [1, 2, 8, 2, 8, 512]    | Fallback |
|  230 | Tensor<[1, 16, 16, 512]> self = ?,<br>List[int] size = [256, 512]              | Fallback |
|  231 | Tensor<[1, 16, 16, 768]> self = ?,<br>List[int] size = [256, 768]              | Fallback |
|  232 | Tensor<[1, 16, 19, 19]> self = ?,<br>List[int] size = [16, 19, 19]             | Done     |
|  233 | Tensor<[1, 16, 19, 64]> self = ?,<br>List[int] size = [16, -1, 64]             | Fallback |
|  234 | Tensor<[1, 16, 197, 197]> self = ?,<br>List[int] size = [16, 197, 197]         | Fallback |
|  235 | Tensor<[1, 16, 197, 64]> self = ?,<br>List[int] size = [16, 197, 64]           | Done     |
|  236 | Tensor<[1, 16, 2, 64]> self = ?,<br>List[int] size = [16, 2, 64]               | Unknown  |
|  237 | Tensor<[1, 16, 256, 256]> self = ?,<br>List[int] size = [16, 256, 256]         | Done     |
|  238 | Tensor<[1, 16, 256, 64]> self = ?,<br>List[int] size = [16, 256, 64]           | Done     |
|  239 | Tensor<[1, 16, 3, 3]> self = ?,<br>List[int] size = [1, -1, 4, 3, 3]           | Fallback |
|  240 | Tensor<[1, 16, 3072]> self = ?,<br>List[int] size = [16, 3072]                 | Done     |
|  241 | Tensor<[1, 16, 32, 32]> self = ?,<br>List[int] size = [16, 32, 32]             | Done     |
|  242 | Tensor<[1, 16, 32, 96]> self = ?,<br>List[int] size = [16, 32, 96]             | Done     |
|  243 | Tensor<[1, 16, 32]> self = ?,<br>List[int] size = [16, 1, 32]                  | Fallback |
|  244 | Tensor<[1, 16, 38, 38]> self = ?,<br>List[int] size = [1, -1, 4, 38, 38]       | Fallback |
|  245 | Tensor<[1, 16, 5, 5]> self = ?,<br>List[int] size = [16, 5, 5]                 | Unknown  |
|  246 | Tensor<[1, 16, 5, 64]> self = ?,<br>List[int] size = [16, 5, 64]               | Unknown  |
|  247 | Tensor<[1, 16, 59, 59]> self = ?,<br>List[int] size = [16, 59, 59]             | Fallback |
|  248 | Tensor<[1, 16, 59, 64]> self = ?,<br>List[int] size = [16, -1, 64]             | Fallback |
|  249 | Tensor<[1, 16, 6, 49, 49]> self = ?,<br>List[int] size = [-1, 6, 49, 49]       | Fallback |
|  250 | Tensor<[1, 16, 6, 64, 64]> self = ?,<br>List[int] size = [-1, 6, 64, 64]       | Fallback |
|  251 | Tensor<[1, 16, 6, 64]> self = ?,<br>List[int] size = [16, 6, 64]               | Unknown  |
|  252 | Tensor<[1, 16, 60, 64]> self = ?,<br>List[int] size = [16, -1, 64]             | Unknown  |
|  253 | Tensor<[1, 16, 64, 10]> self = ?,<br>List[int] size = [16, 64, 10]             | Done     |
|  254 | Tensor<[1, 16, 64, 197]> self = ?,<br>List[int] size = [16, 64, 197]           | Fallback |
|  255 | Tensor<[1, 16, 64, 1]> self = ?,<br>List[int] size = [16, 64, 1]               | Unknown  |
|  256 | Tensor<[1, 16, 64, 256]> self = ?,<br>List[int] size = [16, 64, 256]           | Done     |
|  257 | Tensor<[1, 16, 64, 2]> self = ?,<br>List[int] size = [16, 64, 2]               | Unknown  |
|  258 | Tensor<[1, 16, 64, 5]> self = ?,<br>List[int] size = [16, 64, 5]               | Unknown  |
|  259 | Tensor<[1, 16, 64, 6]> self = ?,<br>List[int] size = [16, 64, 6]               | Unknown  |
|  260 | Tensor<[1, 16, 64, 9]> self = ?,<br>List[int] size = [16, 64, 9]               | Done     |
|  261 | Tensor<[1, 16, 64, s0 + 1]> self = ?,<br>List[int] size = [16, 64, <s0 + 1>]   | Unknown  |
|  262 | Tensor<[1, 16, 64, s10 + 1]> self = ?,<br>List[int] size = [16, 64, <s10 + 1>] | Unknown  |
|  263 | Tensor<[1, 16, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | Fallback |
|  264 | Tensor<[1, 16, 768]> self = ?,<br>List[int] size = [16, 768]                   | Done     |
|  265 | Tensor<[1, 16, 8, 49, 49]> self = ?,<br>List[int] size = [-1, 8, 49, 49]       | Fallback |
|  266 | Tensor<[1, 16, 8, 64, 64]> self = ?,<br>List[int] size = [-1, 8, 64, 64]       | Fallback |
|  267 | Tensor<[1, 16, 9, 128]> self = ?,<br>List[int] size = [16, 9, 128]             | Done     |
|  268 | Tensor<[1, 16, 9, 64]> self = ?,<br>List[int] size = [16, 9, 64]               | Done     |
|  269 | Tensor<[1, 16, 9, 9]> self = ?,<br>List[int] size = [16, 9, 9]                 | Done     |
|  270 | Tensor<[1, 16, 96, 32]> self = ?,<br>List[int] size = [16, 96, 32]             | Done     |
|  271 | Tensor<[1, 16, s0 + 1, 64]> self = ?,<br>List[int] size = [16, <s0 + 1>, 64]   | Unknown  |
|  272 | Tensor<[1, 16, s10 + 1, 64]> self = ?,<br>List[int] size = [16, -1, 64]        | Unknown  |
|  273 | Tensor<[1, 16, s10 + 1, 64]> self = ?,<br>List[int] size = [16, <s10 + 1>, 64] | Unknown  |
|  274 | Tensor<[1, 160, 1024]> self = ?,<br>List[int] size = [1, 160, 32, 32]          | Fallback |
|  275 | Tensor<[1, 160, 16, 16]> self = ?,<br>List[int] size = [1, 160, 256]           | Fallback |
|  276 | Tensor<[1, 160, 32, 32]> self = ?,<br>List[int] size = [1, 160, 1024]          | Fallback |
|  277 | Tensor<[1, 16384, 1, 32]> self = ?,<br>List[int] size = [1, 16384, 32]         | Fallback |
|  278 | Tensor<[1, 16384, 128]> self = ?,<br>List[int] size = [16384, 128]             | Done     |
|  279 | Tensor<[1, 16384, 256]> self = ?,<br>List[int] size = [1, 1, 16384, 256]       | Done     |
|  280 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [1, 1, 16384, 32]         | Done     |
|  281 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [1, 128, 128, -1]         | Fallback |
|  282 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [1, 16384, 1, 32]         | Fallback |
|  283 | Tensor<[1, 16384, 32]> self = ?,<br>List[int] size = [16384, 32]               | Done     |
|  284 | Tensor<[1, 1664, 1, 1]> self = ?,<br>List[int] size = [1, 1664]                | Fallback |
|  285 | Tensor<[1, 16]> self = ?,<br>List[int] size = [1, 1, 1, 16]                    | Done     |
|  286 | Tensor<[1, 19, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]            | Fallback |
|  287 | Tensor<[1, 19, 1024]> self = ?,<br>List[int] size = [1, 19, 16, 64]            | Fallback |
|  288 | Tensor<[1, 19, 1024]> self = ?,<br>List[int] size = [19, 1024]                 | Done     |
|  289 | Tensor<[1, 19, 256008]> self = ?,<br>List[int] size = [-1, 256008]             | Fallback |
|  290 | Tensor<[1, 19, 4096]> self = ?,<br>List[int] size = [19, 4096]                 | Done     |
|  291 | Tensor<[1, 192, 32, 42]> self = ?,<br>List[int] size = [1, 192, 1344]          | Fallback |
|  292 | Tensor<[1, 192, 4150]> self = ?,<br>List[int] size = [1, 192, 50, 83]          | Fallback |
|  293 | Tensor<[1, 1920, 1, 1]> self = ?,<br>List[int] size = [1, 1920]                | Fallback |
|  294 | Tensor<[1, 19200, 1, 64]> self = ?,<br>List[int] size = [1, 19200, 64]         | Fallback |
|  295 | Tensor<[1, 19200, 256]> self = ?,<br>List[int] size = [19200, 256]             | Done     |
|  296 | Tensor<[1, 19200, 300]> self = ?,<br>List[int] size = [1, 1, 19200, 300]       | Fallback |
|  297 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [1, 1, 19200, 64]         | Done     |
|  298 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [1, 120, 160, -1]         | Fallback |
|  299 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [1, 19200, 1, 64]         | Fallback |
|  300 | Tensor<[1, 19200, 64]> self = ?,<br>List[int] size = [19200, 64]               | Done     |
|  301 | Tensor<[1, 196, 3072]> self = ?,<br>List[int] size = [196, 3072]               | Done     |
|  302 | Tensor<[1, 196, 768]> self = ?,<br>List[int] size = [196, 768]                 | Done     |
|  303 | Tensor<[1, 197, 1024]> self = ?,<br>List[int] size = [1, 197, 16, 64]          | Fallback |
|  304 | Tensor<[1, 197, 1024]> self = ?,<br>List[int] size = [197, 1024]               | Done     |
|  305 | Tensor<[1, 197, 12, 64]> self = ?,<br>List[int] size = [1, 197, 768]           | Fallback |
|  306 | Tensor<[1, 197, 16, 64]> self = ?,<br>List[int] size = [1, 197, 1024]          | Fallback |
|  307 | Tensor<[1, 197, 3072]> self = ?,<br>List[int] size = [197, 3072]               | Done     |
|  308 | Tensor<[1, 197, 4096]> self = ?,<br>List[int] size = [197, 4096]               | Done     |
|  309 | Tensor<[1, 197, 768]> self = ?,<br>List[int] size = [1, 197, 12, 64]           | Fallback |
|  310 | Tensor<[1, 197, 768]> self = ?,<br>List[int] size = [197, 768]                 | Done     |
|  311 | Tensor<[1, 19]> self = ?,<br>List[int] size = [-1, 19]                         | Fallback |
|  312 | Tensor<[1, 19]> self = ?,<br>List[int] size = [-1]                             | Fallback |
|  313 | Tensor<[1, 1]> self = ?,<br>List[int] size = [-1, 1]                           | Fallback |
|  314 | Tensor<[1, 1]> self = ?,<br>List[int] size = [-1]                              | Fallback |
|  315 | Tensor<[1, 1]> self = ?,<br>List[int] size = [1]                               | Done     |
|  316 | Tensor<[1, 2, 256, 32]> self = ?,<br>List[int] size = [2, 256, 32]             | Done     |
|  317 | Tensor<[1, 2, 300, 64]> self = ?,<br>List[int] size = [2, 300, 64]             | Done     |
|  318 | Tensor<[1, 2, 32, 256]> self = ?,<br>List[int] size = [2, 32, 256]             | Done     |
|  319 | Tensor<[1, 2, 4096, 256]> self = ?,<br>List[int] size = [2, 4096, 256]         | Done     |
|  320 | Tensor<[1, 2, 4096, 32]> self = ?,<br>List[int] size = [2, 4096, 32]           | Done     |
|  321 | Tensor<[1, 2, 4800, 300]> self = ?,<br>List[int] size = [2, 4800, 300]         | Fallback |
|  322 | Tensor<[1, 2, 4800, 64]> self = ?,<br>List[int] size = [2, 4800, 64]           | Done     |
|  323 | Tensor<[1, 2, 64, 300]> self = ?,<br>List[int] size = [2, 64, 300]             | Fallback |
|  324 | Tensor<[1, 201, 12, 64]> self = ?,<br>List[int] size = [1, 201, 768]           | Unknown  |
|  325 | Tensor<[1, 201, 3072]> self = ?,<br>List[int] size = [201, 3072]               | Unknown  |
|  326 | Tensor<[1, 201, 768]> self = ?,<br>List[int] size = [1, 201, 12, 64]           | Unknown  |
|  327 | Tensor<[1, 201, 768]> self = ?,<br>List[int] size = [201, 768]                 | Unknown  |
|  328 | Tensor<[1, 2016, 1, 1]> self = ?,<br>List[int] size = [1, 2016]                | Fallback |
|  329 | Tensor<[1, 2048, 1, 1]> self = ?,<br>List[int] size = [1, 2048]                | Fallback |
|  330 | Tensor<[1, 2048, 1280]> self = ?,<br>List[int] size = [1, 2048, 8, 160]        | Unknown  |
|  331 | Tensor<[1, 2048, 15, 20]> self = ?,<br>List[int] size = [1, 2048, 300]         | Fallback |
|  332 | Tensor<[1, 2048, 256]> self = ?,<br>List[int] size = [1, 2048, 8, 32]          | Unknown  |
|  333 | Tensor<[1, 2048, 300]> self = ?,<br>List[int] size = [1, 2048, 15, 20]         | Fallback |
|  334 | Tensor<[1, 2048, 768]> self = ?,<br>List[int] size = [-1, 768]                 | Unknown  |
|  335 | Tensor<[1, 2048, 768]> self = ?,<br>List[int] size = [2048, 768]               | Unknown  |
|  336 | Tensor<[1, 2048, 8, 96]> self = ?,<br>List[int] size = [1, 2048, 768]          | Unknown  |
|  337 | Tensor<[1, 2048]> self = ?,<br>List[int] size = [1, 1, 2048]                   | Unknown  |
|  338 | Tensor<[1, 2208, 1, 1]> self = ?,<br>List[int] size = [1, 2208]                | Fallback |
|  339 | Tensor<[1, 23, 40, 64, 2]> self = ?,<br>List[int] size = [1, 23, 40, 128]      | Unknown  |
|  340 | Tensor<[1, 23, 40]> self = ?,<br>List[int] size = [1, 920]                     | Unknown  |
|  341 | Tensor<[1, 24, 1, 1]> self = ?,<br>List[int] size = [1, -1, 4, 1, 1]           | Fallback |
|  342 | Tensor<[1, 24, 10, 10]> self = ?,<br>List[int] size = [1, -1, 4, 10, 10]       | Fallback |
|  343 | Tensor<[1, 24, 19, 19]> self = ?,<br>List[int] size = [1, -1, 4, 19, 19]       | Fallback |
|  344 | Tensor<[1, 24, 2, 2]> self = ?,<br>List[int] size = [1, -1, 4, 2, 2]           | Fallback |
|  345 | Tensor<[1, 24, 20, 20]> self = ?,<br>List[int] size = [1, -1, 4, 20, 20]       | Fallback |
|  346 | Tensor<[1, 24, 3, 3]> self = ?,<br>List[int] size = [1, -1, 4, 3, 3]           | Fallback |
|  347 | Tensor<[1, 24, 3072]> self = ?,<br>List[int] size = [24, 3072]                 | Done     |
|  348 | Tensor<[1, 24, 32, 49]> self = ?,<br>List[int] size = [24, 32, 49]             | Fallback |
|  349 | Tensor<[1, 24, 32, 64]> self = ?,<br>List[int] size = [24, 32, 64]             | Done     |
|  350 | Tensor<[1, 24, 49, 32]> self = ?,<br>List[int] size = [24, 49, 32]             | Done     |
|  351 | Tensor<[1, 24, 49, 49]> self = ?,<br>List[int] size = [24, 49, 49]             | Fallback |
|  352 | Tensor<[1, 24, 5, 5]> self = ?,<br>List[int] size = [1, -1, 4, 5, 5]           | Fallback |
|  353 | Tensor<[1, 24, 64, 32]> self = ?,<br>List[int] size = [24, 64, 32]             | Done     |
|  354 | Tensor<[1, 24, 64, 64]> self = ?,<br>List[int] size = [24, 64, 64]             | Done     |
|  355 | Tensor<[1, 24, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | Fallback |
|  356 | Tensor<[1, 24, 768]> self = ?,<br>List[int] size = [1, 24, 12, 64]             | Fallback |
|  357 | Tensor<[1, 24, 768]> self = ?,<br>List[int] size = [24, 768]                   | Done     |
|  358 | Tensor<[1, 25, 12, 64]> self = ?,<br>List[int] size = [1, 25, 768]             | Fallback |
|  359 | Tensor<[1, 25, 3072]> self = ?,<br>List[int] size = [25, 3072]                 | Done     |
|  360 | Tensor<[1, 25, 768]> self = ?,<br>List[int] size = [1, 25, 12, 64]             | Fallback |
|  361 | Tensor<[1, 25, 768]> self = ?,<br>List[int] size = [25, 768]                   | Done     |
|  362 | Tensor<[1, 2520, 1, 1]> self = ?,<br>List[int] size = [1, 2520]                | Fallback |
|  363 | Tensor<[1, 255, 16, 16]> self = ?,<br>List[int] size = [1, 3, 85, 16, 16]      | Unknown  |
|  364 | Tensor<[1, 255, 32, 32]> self = ?,<br>List[int] size = [1, 3, 85, 32, 32]      | Unknown  |
|  365 | Tensor<[1, 255, 64, 64]> self = ?,<br>List[int] size = [1, 3, 85, 64, 64]      | Unknown  |
|  366 | Tensor<[1, 256, 1024]> self = ?,<br>List[int] size = [1, 256, 16, 64]          | Fallback |
|  367 | Tensor<[1, 256, 1024]> self = ?,<br>List[int] size = [1, 256, 32, 32]          | Fallback |
|  368 | Tensor<[1, 256, 1024]> self = ?,<br>List[int] size = [256, 1024]               | Done     |
|  369 | Tensor<[1, 256, 120, 160]> self = ?,<br>List[int] size = [1, 256, 19200]       | Fallback |
|  370 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [1, -1, 8, 160]           | Fallback |
|  371 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [1, 16, 16, 1280]         | Fallback |
|  372 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [1, 256, 8, 160]          | Unknown  |
|  373 | Tensor<[1, 256, 1280]> self = ?,<br>List[int] size = [256, 1280]               | Done     |
|  374 | Tensor<[1, 256, 16, 16]> self = ?,<br>List[int] size = [1, 256, 256]           | Fallback |
|  375 | Tensor<[1, 256, 16, 64]> self = ?,<br>List[int] size = [1, 256, 1024]          | Fallback |
|  376 | Tensor<[1, 256, 160]> self = ?,<br>List[int] size = [1, 256, 5, 32]            | Fallback |
|  377 | Tensor<[1, 256, 160]> self = ?,<br>List[int] size = [256, 160]                 | Done     |
|  378 | Tensor<[1, 256, 16384]> self = ?,<br>List[int] size = [1, 256, 128, 128]       | Fallback |
|  379 | Tensor<[1, 256, 19200]> self = ?,<br>List[int] size = [1, 256, 120, 160]       | Fallback |
|  380 | Tensor<[1, 256, 23, 40]> self = ?,<br>List[int] size = [1, 256, 920]           | Unknown  |
|  381 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [1, 16, 16, -1]            | Fallback |
|  382 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [1, 256, 16, 16]           | Fallback |
|  383 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [1, 256, 8, 32]            | Fallback |
|  384 | Tensor<[1, 256, 256]> self = ?,<br>List[int] size = [256, 256]                 | Done     |
|  385 | Tensor<[1, 256, 32]> self = ?,<br>List[int] size = [1, 256, 1, 32]             | Fallback |
|  386 | Tensor<[1, 256, 32]> self = ?,<br>List[int] size = [256, 32]                   | Done     |
|  387 | Tensor<[1, 256, 4096]> self = ?,<br>List[int] size = [1, 256, 64, 64]          | Fallback |
|  388 | Tensor<[1, 256, 4096]> self = ?,<br>List[int] size = [256, 4096]               | Done     |
|  389 | Tensor<[1, 256, 5120]> self = ?,<br>List[int] size = [256, 5120]               | Done     |
|  390 | Tensor<[1, 256, 512]> self = ?,<br>List[int] size = [1, 256, 512]              | Done     |
|  391 | Tensor<[1, 256, 512]> self = ?,<br>List[int] size = [256, 512]                 | Done     |
|  392 | Tensor<[1, 256, 64, 64]> self = ?,<br>List[int] size = [1, 256, 4096]          | Fallback |
|  393 | Tensor<[1, 256, 64]> self = ?,<br>List[int] size = [1, 256, 2, 32]             | Fallback |
|  394 | Tensor<[1, 256, 64]> self = ?,<br>List[int] size = [256, 64]                   | Done     |
|  395 | Tensor<[1, 256, 768]> self = ?,<br>List[int] size = [1, 256, 8, 96]            | Unknown  |
|  396 | Tensor<[1, 256, 768]> self = ?,<br>List[int] size = [256, 768]                 | Done     |
|  397 | Tensor<[1, 256, 8, 160]> self = ?,<br>List[int] size = [1, -1, 1280]           | Fallback |
|  398 | Tensor<[1, 256, 8, 160]> self = ?,<br>List[int] size = [1, 256, 1280]          | Unknown  |
|  399 | Tensor<[1, 256, 8, 32]> self = ?,<br>List[int] size = [1, 256, 256]            | Fallback |
|  400 | Tensor<[1, 256]> self = ?,<br>List[int] size = [1, 1, 256]                     | Unknown  |
|  401 | Tensor<[1, 25]> self = ?,<br>List[int] size = [1, 25]                          | Done     |
|  402 | Tensor<[1, 28, 28, 1024]> self = ?,<br>List[int] size = [784, 1024]            | Fallback |
|  403 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] size = [1, 4, 7, 4, 7, 192]    | Fallback |
|  404 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] size = [784, 192]              | Fallback |
|  405 | Tensor<[1, 28, 28, 256]> self = ?,<br>List[int] size = [1, 4, 7, 4, 7, 256]    | Fallback |
|  406 | Tensor<[1, 28, 28, 256]> self = ?,<br>List[int] size = [784, 256]              | Fallback |
|  407 | Tensor<[1, 28, 28, 384]> self = ?,<br>List[int] size = [784, 384]              | Fallback |
|  408 | Tensor<[1, 28, 28, 512]> self = ?,<br>List[int] size = [784, 512]              | Fallback |
|  409 | Tensor<[1, 28, 28, 768]> self = ?,<br>List[int] size = [784, 768]              | Fallback |
|  410 | Tensor<[1, 3, 1445, 1445]> self = ?,<br>List[int] size = [3, 1445, 1445]       | Fallback |
|  411 | Tensor<[1, 3, 1445, 64]> self = ?,<br>List[int] size = [3, 1445, 64]           | Done     |
|  412 | Tensor<[1, 3, 16, 16, 85]> self = ?,<br>List[int] size = [1, 768, 85]          | Unknown  |
|  413 | Tensor<[1, 3, 256, 256]> self = ?,<br>List[int] size = [1, 3, 16, 16, 16, 16]  | Fallback |
|  414 | Tensor<[1, 3, 32, 32, 85]> self = ?,<br>List[int] size = [1, 3072, 85]         | Unknown  |
|  415 | Tensor<[1, 3, 64, 1445]> self = ?,<br>List[int] size = [3, 64, 1445]           | Fallback |
|  416 | Tensor<[1, 3, 64, 64, 85]> self = ?,<br>List[int] size = [1, 12288, 85]        | Unknown  |
|  417 | Tensor<[1, 300, 128]> self = ?,<br>List[int] size = [1, 300, 2, 64]            | Fallback |
|  418 | Tensor<[1, 300, 128]> self = ?,<br>List[int] size = [300, 128]                 | Done     |
|  419 | Tensor<[1, 300, 2048]> self = ?,<br>List[int] size = [300, 2048]               | Done     |
|  420 | Tensor<[1, 300, 320]> self = ?,<br>List[int] size = [1, 300, 5, 64]            | Fallback |
|  421 | Tensor<[1, 300, 320]> self = ?,<br>List[int] size = [300, 320]                 | Done     |
|  422 | Tensor<[1, 300, 512]> self = ?,<br>List[int] size = [1, 15, 20, -1]            | Fallback |
|  423 | Tensor<[1, 300, 512]> self = ?,<br>List[int] size = [1, 300, 8, 64]            | Fallback |
|  424 | Tensor<[1, 300, 512]> self = ?,<br>List[int] size = [300, 512]                 | Done     |
|  425 | Tensor<[1, 300, 64]> self = ?,<br>List[int] size = [1, 300, 1, 64]             | Fallback |
|  426 | Tensor<[1, 300, 64]> self = ?,<br>List[int] size = [300, 64]                   | Done     |
|  427 | Tensor<[1, 300, 8, 64]> self = ?,<br>List[int] size = [1, 300, 512]            | Fallback |
|  428 | Tensor<[1, 3024, 1, 1]> self = ?,<br>List[int] size = [1, 3024]                | Fallback |
|  429 | Tensor<[1, 3072]> self = ?,<br>List[int] size = [1, 1, 3072]                   | Done     |
|  430 | Tensor<[1, 32, 128, 128]> self = ?,<br>List[int] size = [1, 32, 16384]         | Fallback |
|  431 | Tensor<[1, 32, 1536]> self = ?,<br>List[int] size = [32, 1536]                 | Done     |
|  432 | Tensor<[1, 32, 16, 16]> self = ?,<br>List[int] size = [1, 32, 256]             | Fallback |
|  433 | Tensor<[1, 32, 16384]> self = ?,<br>List[int] size = [1, 32, 128, 128]         | Fallback |
|  434 | Tensor<[1, 32, 32, 1024]> self = ?,<br>List[int] size = [1024, 1024]           | Fallback |
|  435 | Tensor<[1, 32, 32, 192]> self = ?,<br>List[int] size = [1, 4, 8, 4, 8, 192]    | Fallback |
|  436 | Tensor<[1, 32, 32, 192]> self = ?,<br>List[int] size = [1024, 192]             | Fallback |
|  437 | Tensor<[1, 32, 32, 256]> self = ?,<br>List[int] size = [1, 4, 8, 4, 8, 256]    | Fallback |
|  438 | Tensor<[1, 32, 32, 256]> self = ?,<br>List[int] size = [1024, 256]             | Fallback |
|  439 | Tensor<[1, 32, 32, 384]> self = ?,<br>List[int] size = [1024, 384]             | Fallback |
|  440 | Tensor<[1, 32, 32, 49]> self = ?,<br>List[int] size = [32, 32, 49]             | Fallback |
|  441 | Tensor<[1, 32, 32, 512]> self = ?,<br>List[int] size = [1024, 512]             | Fallback |
|  442 | Tensor<[1, 32, 32, 640]> self = ?,<br>List[int] size = [1, 1024, 640]          | Fallback |
|  443 | Tensor<[1, 32, 32, 64]> self = ?,<br>List[int] size = [32, 32, 64]             | Done     |
|  444 | Tensor<[1, 32, 32, 768]> self = ?,<br>List[int] size = [1024, 768]             | Fallback |
|  445 | Tensor<[1, 32, 4608]> self = ?,<br>List[int] size = [1, 32, 16, 3, 96]         | Fallback |
|  446 | Tensor<[1, 32, 49, 32]> self = ?,<br>List[int] size = [32, 49, 32]             | Done     |
|  447 | Tensor<[1, 32, 49, 49]> self = ?,<br>List[int] size = [32, 49, 49]             | Fallback |
|  448 | Tensor<[1, 32, 6144]> self = ?,<br>List[int] size = [32, 6144]                 | Done     |
|  449 | Tensor<[1, 32, 64, 32]> self = ?,<br>List[int] size = [32, 64, 32]             | Done     |
|  450 | Tensor<[1, 32, 64, 64]> self = ?,<br>List[int] size = [32, 64, 64]             | Done     |
|  451 | Tensor<[1, 320, 1200]> self = ?,<br>List[int] size = [1, 320, 30, 40]          | Fallback |
|  452 | Tensor<[1, 320, 15, 20]> self = ?,<br>List[int] size = [1, 320, 300]           | Fallback |
|  453 | Tensor<[1, 320, 30, 40]> self = ?,<br>List[int] size = [1, 320, 1200]          | Fallback |
|  454 | Tensor<[1, 32128]> self = ?,<br>List[int] size = [1, 1, 32128]                 | Unknown  |
|  455 | Tensor<[1, 36, 100, 136]> self = ?,<br>List[int] size = [1, -1, 4, 100, 136]   | Unknown  |
|  456 | Tensor<[1, 36, 13, 17]> self = ?,<br>List[int] size = [1, -1, 4, 13, 17]       | Unknown  |
|  457 | Tensor<[1, 36, 25, 34]> self = ?,<br>List[int] size = [1, -1, 4, 25, 34]       | Unknown  |
|  458 | Tensor<[1, 36, 50, 68]> self = ?,<br>List[int] size = [1, -1, 4, 50, 68]       | Unknown  |
|  459 | Tensor<[1, 36, 7, 9]> self = ?,<br>List[int] size = [1, -1, 4, 7, 9]           | Unknown  |
|  460 | Tensor<[1, 364, 1, 1]> self = ?,<br>List[int] size = [1, -1, 91, 1, 1]         | Fallback |
|  461 | Tensor<[1, 364, 3, 3]> self = ?,<br>List[int] size = [1, -1, 91, 3, 3]         | Fallback |
|  462 | Tensor<[1, 364, 38, 38]> self = ?,<br>List[int] size = [1, -1, 91, 38, 38]     | Fallback |
|  463 | Tensor<[1, 3712, 1, 1]> self = ?,<br>List[int] size = [1, 3712]                | Fallback |
|  464 | Tensor<[1, 384]> self = ?,<br>List[int] size = [1, 1, 384]                     | Unknown  |
|  465 | Tensor<[1, 4, 12, 49, 49]> self = ?,<br>List[int] size = [-1, 12, 49, 49]      | Fallback |
|  466 | Tensor<[1, 4, 12, 64, 64]> self = ?,<br>List[int] size = [-1, 12, 64, 64]      | Fallback |
|  467 | Tensor<[1, 4, 12, 64]> self = ?,<br>List[int] size = [1, 4, 768]               | Unknown  |
|  468 | Tensor<[1, 4, 16, 49, 49]> self = ?,<br>List[int] size = [-1, 16, 49, 49]      | Fallback |
|  469 | Tensor<[1, 4, 16, 64, 64]> self = ?,<br>List[int] size = [-1, 16, 64, 64]      | Fallback |
|  470 | Tensor<[1, 4, 3072]> self = ?,<br>List[int] size = [4, 3072]                   | Unknown  |
|  471 | Tensor<[1, 4, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]              | Unknown  |
|  472 | Tensor<[1, 4, 768]> self = ?,<br>List[int] size = [1, 4, 12, 64]               | Unknown  |
|  473 | Tensor<[1, 4, 768]> self = ?,<br>List[int] size = [4, 768]                     | Unknown  |
|  474 | Tensor<[1, 400, 1, 1]> self = ?,<br>List[int] size = [1, 400]                  | Fallback |
|  475 | Tensor<[1, 4096, 1280]> self = ?,<br>List[int] size = [4096, 1280]             | Done     |
|  476 | Tensor<[1, 4096, 2, 32]> self = ?,<br>List[int] size = [1, 4096, 64]           | Fallback |
|  477 | Tensor<[1, 4096, 256]> self = ?,<br>List[int] size = [4096, 256]               | Done     |
|  478 | Tensor<[1, 4096, 320]> self = ?,<br>List[int] size = [1, -1, 8, 40]            | Fallback |
|  479 | Tensor<[1, 4096, 320]> self = ?,<br>List[int] size = [1, 64, 64, 320]          | Fallback |
|  480 | Tensor<[1, 4096, 320]> self = ?,<br>List[int] size = [4096, 320]               | Done     |
|  481 | Tensor<[1, 4096, 64]> self = ?,<br>List[int] size = [1, 4096, 2, 32]           | Fallback |
|  482 | Tensor<[1, 4096, 64]> self = ?,<br>List[int] size = [1, 64, 64, -1]            | Fallback |
|  483 | Tensor<[1, 4096, 64]> self = ?,<br>List[int] size = [4096, 64]                 | Done     |
|  484 | Tensor<[1, 4096, 8, 40]> self = ?,<br>List[int] size = [1, -1, 320]            | Fallback |
|  485 | Tensor<[1, 4096]> self = ?,<br>List[int] size = [1, 1, 4096]                   | Unknown  |
|  486 | Tensor<[1, 440, 1, 1]> self = ?,<br>List[int] size = [1, 440]                  | Fallback |
|  487 | Tensor<[1, 45, 12, 64]> self = ?,<br>List[int] size = [1, 45, 768]             | Unknown  |
|  488 | Tensor<[1, 45, 3072]> self = ?,<br>List[int] size = [45, 3072]                 | Unknown  |
|  489 | Tensor<[1, 45, 768]> self = ?,<br>List[int] size = [-1, 45, 768]               | Unknown  |
|  490 | Tensor<[1, 45, 768]> self = ?,<br>List[int] size = [1, 45, 12, 64]             | Unknown  |
|  491 | Tensor<[1, 45, 768]> self = ?,<br>List[int] size = [45, 768]                   | Unknown  |
|  492 | Tensor<[1, 45]> self = ?,<br>List[int] size = [-1, 45]                         | Unknown  |
|  493 | Tensor<[1, 4800, 128]> self = ?,<br>List[int] size = [1, 4800, 2, 64]          | Fallback |
|  494 | Tensor<[1, 4800, 128]> self = ?,<br>List[int] size = [1, 60, 80, -1]           | Fallback |
|  495 | Tensor<[1, 4800, 128]> self = ?,<br>List[int] size = [4800, 128]               | Done     |
|  496 | Tensor<[1, 4800, 2, 64]> self = ?,<br>List[int] size = [1, 4800, 128]          | Fallback |
|  497 | Tensor<[1, 4800, 512]> self = ?,<br>List[int] size = [4800, 512]               | Done     |
|  498 | Tensor<[1, 49, 1024]> self = ?,<br>List[int] size = [1, 1, 1, 7, 7, 1024]      | Fallback |
|  499 | Tensor<[1, 49, 1024]> self = ?,<br>List[int] size = [49, 1024]                 | Done     |
|  500 | Tensor<[1, 49, 2304]> self = ?,<br>List[int] size = [1, 49, 3, 24, 32]         | Fallback |
|  501 | Tensor<[1, 49, 3072]> self = ?,<br>List[int] size = [1, 49, 3, 32, 32]         | Fallback |
|  502 | Tensor<[1, 49, 768]> self = ?,<br>List[int] size = [1, 1, 1, 7, 7, 768]        | Fallback |
|  503 | Tensor<[1, 49, 768]> self = ?,<br>List[int] size = [49, 768]                   | Done     |
|  504 | Tensor<[1, 4]> self = ?,<br>List[int] size = [-1, 4]                           | Unknown  |
|  505 | Tensor<[1, 5, 1, 16, 2]> self = ?,<br>List[int] size = [1, 5, 1, 32]           | Unknown  |
|  506 | Tensor<[1, 5, 1024, 256]> self = ?,<br>List[int] size = [5, 1024, 256]         | Done     |
|  507 | Tensor<[1, 5, 1024, 32]> self = ?,<br>List[int] size = [5, 1024, 32]           | Done     |
|  508 | Tensor<[1, 5, 1024]> self = ?,<br>List[int] size = [1, 5, 1024]                | Unknown  |
|  509 | Tensor<[1, 5, 1024]> self = ?,<br>List[int] size = [5, 1024]                   | Unknown  |
|  510 | Tensor<[1, 5, 1200, 300]> self = ?,<br>List[int] size = [5, 1200, 300]         | Fallback |
|  511 | Tensor<[1, 5, 1200, 64]> self = ?,<br>List[int] size = [5, 1200, 64]           | Done     |
|  512 | Tensor<[1, 5, 16, 16, 2]> self = ?,<br>List[int] size = [1, 5, 16, 32]         | Unknown  |
|  513 | Tensor<[1, 5, 16, 64]> self = ?,<br>List[int] size = [1, 5, 1024]              | Unknown  |
|  514 | Tensor<[1, 5, 256, 32]> self = ?,<br>List[int] size = [5, 256, 32]             | Done     |
|  515 | Tensor<[1, 5, 300, 64]> self = ?,<br>List[int] size = [5, 300, 64]             | Done     |
|  516 | Tensor<[1, 5, 3072]> self = ?,<br>List[int] size = [1, 5, 4, -1]               | Unknown  |
|  517 | Tensor<[1, 5, 32, 256]> self = ?,<br>List[int] size = [5, 32, 256]             | Done     |
|  518 | Tensor<[1, 5, 4, 256]> self = ?,<br>List[int] size = [1, 5, 4, 4, 64]          | Unknown  |
|  519 | Tensor<[1, 5, 4096]> self = ?,<br>List[int] size = [5, 4096]                   | Unknown  |
|  520 | Tensor<[1, 5, 64, 300]> self = ?,<br>List[int] size = [5, 64, 300]             | Fallback |
|  521 | Tensor<[1, 50, 1024]> self = ?,<br>List[int] size = [50, 1024]                 | Done     |
|  522 | Tensor<[1, 50, 3072]> self = ?,<br>List[int] size = [50, 3072]                 | Done     |
|  523 | Tensor<[1, 50, 4096]> self = ?,<br>List[int] size = [50, 4096]                 | Done     |
|  524 | Tensor<[1, 50, 768]> self = ?,<br>List[int] size = [1, -1, 12, 64]             | Fallback |
|  525 | Tensor<[1, 50, 768]> self = ?,<br>List[int] size = [1, 50, 12, 64]             | Fallback |
|  526 | Tensor<[1, 50, 768]> self = ?,<br>List[int] size = [50, 768]                   | Done     |
|  527 | Tensor<[1, 50257]> self = ?,<br>List[int] size = [1, 1, 50257]                 | Unknown  |
|  528 | Tensor<[1, 50272]> self = ?,<br>List[int] size = [1, 1, 50272]                 | Unknown  |
|  529 | Tensor<[1, 512, 1, 1]> self = ?,<br>List[int] size = [1, 512]                  | Fallback |
|  530 | Tensor<[1, 512, 15, 20]> self = ?,<br>List[int] size = [1, 512, 300]           | Fallback |
|  531 | Tensor<[1, 512, 4800]> self = ?,<br>List[int] size = [1, 512, 60, 80]          | Fallback |
|  532 | Tensor<[1, 512, 60, 80]> self = ?,<br>List[int] size = [1, 512, 4800]          | Fallback |
|  533 | Tensor<[1, 512, 7, 7]> self = ?,<br>List[int] size = [1, 25088]                | Fallback |
|  534 | Tensor<[1, 51200]> self = ?,<br>List[int] size = [1, 1, 51200]                 | Unknown  |
|  535 | Tensor<[1, 512]> self = ?,<br>List[int] size = [1, 1, 512]                     | Unknown  |
|  536 | Tensor<[1, 512]> self = ?,<br>List[int] size = [1, 512]                        | Done     |
|  537 | Tensor<[1, 51865]> self = ?,<br>List[int] size = [1, 1, 51865]                 | Fallback |
|  538 | Tensor<[1, 546, 1, 1]> self = ?,<br>List[int] size = [1, -1, 91, 1, 1]         | Fallback |
|  539 | Tensor<[1, 546, 10, 10]> self = ?,<br>List[int] size = [1, -1, 91, 10, 10]     | Fallback |
|  540 | Tensor<[1, 546, 19, 19]> self = ?,<br>List[int] size = [1, -1, 91, 19, 19]     | Fallback |
|  541 | Tensor<[1, 546, 2, 2]> self = ?,<br>List[int] size = [1, -1, 91, 2, 2]         | Fallback |
|  542 | Tensor<[1, 546, 20, 20]> self = ?,<br>List[int] size = [1, -1, 91, 20, 20]     | Fallback |
|  543 | Tensor<[1, 546, 3, 3]> self = ?,<br>List[int] size = [1, -1, 91, 3, 3]         | Fallback |
|  544 | Tensor<[1, 546, 5, 5]> self = ?,<br>List[int] size = [1, -1, 91, 5, 5]         | Fallback |
|  545 | Tensor<[1, 56, 56, 128]> self = ?,<br>List[int] size = [1, 8, 7, 8, 7, 128]    | Fallback |
|  546 | Tensor<[1, 56, 56, 128]> self = ?,<br>List[int] size = [3136, 128]             | Fallback |
|  547 | Tensor<[1, 56, 56, 384]> self = ?,<br>List[int] size = [3136, 384]             | Fallback |
|  548 | Tensor<[1, 56, 56, 512]> self = ?,<br>List[int] size = [3136, 512]             | Fallback |
|  549 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] size = [1, 8, 7, 8, 7, 96]      | Fallback |
|  550 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] size = [3136, 96]               | Fallback |
|  551 | Tensor<[1, 576, 1, 1]> self = ?,<br>List[int] size = [1, 576]                  | Fallback |
|  552 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [-1, 1024]                 | Fallback |
|  553 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [1, -1, 16, 64]            | Fallback |
|  554 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [1, 59, 16, 64]            | Fallback |
|  555 | Tensor<[1, 59, 1024]> self = ?,<br>List[int] size = [59, 1024]                 | Done     |
|  556 | Tensor<[1, 59, 512]> self = ?,<br>List[int] size = [59, 512]                   | Done     |
|  557 | Tensor<[1, 59]> self = ?,<br>List[int] size = [-1, 59]                         | Fallback |
|  558 | Tensor<[1, 5]> self = ?,<br>List[int] size = [-1, 5]                           | Unknown  |
|  559 | Tensor<[1, 5]> self = ?,<br>List[int] size = [1, -1]                           | Unknown  |
|  560 | Tensor<[1, 6, 1, 15]> self = ?,<br>List[int] size = [6, 1, 15]                 | Unknown  |
|  561 | Tensor<[1, 6, 1, 17]> self = ?,<br>List[int] size = [6, 1, 17]                 | Unknown  |
|  562 | Tensor<[1, 6, 1, 1]> self = ?,<br>List[int] size = [6, 1, 1]                   | Unknown  |
|  563 | Tensor<[1, 6, 1, 2]> self = ?,<br>List[int] size = [6, 1, 2]                   | Unknown  |
|  564 | Tensor<[1, 6, 1, 64]> self = ?,<br>List[int] size = [6, 1, 64]                 | Unknown  |
|  565 | Tensor<[1, 6, 1, s0 + 1]> self = ?,<br>List[int] size = [6, 1, <s0 + 1>]       | Unknown  |
|  566 | Tensor<[1, 6, 15, 15]> self = ?,<br>List[int] size = [6, 15, 15]               | Done     |
|  567 | Tensor<[1, 6, 15, 64]> self = ?,<br>List[int] size = [6, 15, 64]               | Done     |
|  568 | Tensor<[1, 6, 17, 64]> self = ?,<br>List[int] size = [6, 17, 64]               | Unknown  |
|  569 | Tensor<[1, 6, 2, 64]> self = ?,<br>List[int] size = [6, 2, 64]                 | Unknown  |
|  570 | Tensor<[1, 6, 64, 15]> self = ?,<br>List[int] size = [6, 64, 15]               | Done     |
|  571 | Tensor<[1, 6, 64, 17]> self = ?,<br>List[int] size = [6, 64, 17]               | Unknown  |
|  572 | Tensor<[1, 6, 64, 1]> self = ?,<br>List[int] size = [6, 64, 1]                 | Unknown  |
|  573 | Tensor<[1, 6, 64, 2]> self = ?,<br>List[int] size = [6, 64, 2]                 | Unknown  |
|  574 | Tensor<[1, 6, 64, s0 + 1]> self = ?,<br>List[int] size = [6, 64, <s0 + 1>]     | Unknown  |
|  575 | Tensor<[1, 6, s0 + 1, 64]> self = ?,<br>List[int] size = [6, <s0 + 1>, 64]     | Unknown  |
|  576 | Tensor<[1, 64, 1024]> self = ?,<br>List[int] size = [1, 1, 1, 8, 8, 1024]      | Fallback |
|  577 | Tensor<[1, 64, 1024]> self = ?,<br>List[int] size = [64, 1024]                 | Done     |
|  578 | Tensor<[1, 64, 12, 12]> self = ?,<br>List[int] size = [1, 9216]                | Fallback |
|  579 | Tensor<[1, 64, 120, 160]> self = ?,<br>List[int] size = [1, 64, 19200]         | Fallback |
|  580 | Tensor<[1, 64, 1280]> self = ?,<br>List[int] size = [1, -1, 8, 160]            | Fallback |
|  581 | Tensor<[1, 64, 1280]> self = ?,<br>List[int] size = [1, 8, 8, 1280]            | Fallback |
|  582 | Tensor<[1, 64, 1280]> self = ?,<br>List[int] size = [64, 1280]                 | Done     |
|  583 | Tensor<[1, 64, 15, 20]> self = ?,<br>List[int] size = [1, 64, 300]             | Fallback |
|  584 | Tensor<[1, 64, 16, 16]> self = ?,<br>List[int] size = [1, 64, 256]             | Fallback |
|  585 | Tensor<[1, 64, 19200]> self = ?,<br>List[int] size = [1, 64, 120, 160]         | Fallback |
|  586 | Tensor<[1, 64, 2304]> self = ?,<br>List[int] size = [1, 64, 3, 24, 32]         | Fallback |
|  587 | Tensor<[1, 64, 3, 49, 49]> self = ?,<br>List[int] size = [-1, 3, 49, 49]       | Fallback |
|  588 | Tensor<[1, 64, 3, 64, 64]> self = ?,<br>List[int] size = [-1, 3, 64, 64]       | Fallback |
|  589 | Tensor<[1, 64, 3072]> self = ?,<br>List[int] size = [1, 64, 3, 32, 32]         | Fallback |
|  590 | Tensor<[1, 64, 4, 49, 49]> self = ?,<br>List[int] size = [-1, 4, 49, 49]       | Fallback |
|  591 | Tensor<[1, 64, 4, 64, 64]> self = ?,<br>List[int] size = [-1, 4, 64, 64]       | Fallback |
|  592 | Tensor<[1, 64, 4096]> self = ?,<br>List[int] size = [1, 64, 64, 64]            | Fallback |
|  593 | Tensor<[1, 64, 5120]> self = ?,<br>List[int] size = [64, 5120]                 | Done     |
|  594 | Tensor<[1, 64, 64, 128]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 128]    | Fallback |
|  595 | Tensor<[1, 64, 64, 128]> self = ?,<br>List[int] size = [4096, 128]             | Fallback |
|  596 | Tensor<[1, 64, 64, 320]> self = ?,<br>List[int] size = [1, 4096, 320]          | Fallback |
|  597 | Tensor<[1, 64, 64, 384]> self = ?,<br>List[int] size = [4096, 384]             | Fallback |
|  598 | Tensor<[1, 64, 64, 512]> self = ?,<br>List[int] size = [4096, 512]             | Fallback |
|  599 | Tensor<[1, 64, 64, 64]> self = ?,<br>List[int] size = [1, 64, 4096]            | Fallback |
|  600 | Tensor<[1, 64, 64, 96]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 96]      | Fallback |
|  601 | Tensor<[1, 64, 64, 96]> self = ?,<br>List[int] size = [4096, 96]               | Fallback |
|  602 | Tensor<[1, 64, 64, 9]> self = ?,<br>List[int] size = [64, 64, 9]               | Done     |
|  603 | Tensor<[1, 64, 768]> self = ?,<br>List[int] size = [1, 1, 1, 8, 8, 768]        | Fallback |
|  604 | Tensor<[1, 64, 768]> self = ?,<br>List[int] size = [64, 768]                   | Done     |
|  605 | Tensor<[1, 64, 8, 160]> self = ?,<br>List[int] size = [1, -1, 1280]            | Fallback |
|  606 | Tensor<[1, 64, 9, 64]> self = ?,<br>List[int] size = [64, 9, 64]               | Done     |
|  607 | Tensor<[1, 64, 9, 9]> self = ?,<br>List[int] size = [64, 9, 9]                 | Done     |
|  608 | Tensor<[1, 640, 1024]> self = ?,<br>List[int] size = [1, 640, 32, 32]          | Fallback |
|  609 | Tensor<[1, 640, 32, 32]> self = ?,<br>List[int] size = [1, 640, 1024]          | Fallback |
|  610 | Tensor<[1, 672, 1, 1]> self = ?,<br>List[int] size = [1, 672]                  | Fallback |
|  611 | Tensor<[1, 6]> self = ?,<br>List[int] size = [1, -1]                           | Unknown  |
|  612 | Tensor<[1, 7, 12, 64]> self = ?,<br>List[int] size = [1, 7, 768]               | Unknown  |
|  613 | Tensor<[1, 7, 18176]> self = ?,<br>List[int] size = [7, 18176]                 | Unknown  |
|  614 | Tensor<[1, 7, 3072]> self = ?,<br>List[int] size = [-1, 3072]                  | Unknown  |
|  615 | Tensor<[1, 7, 4544]> self = ?,<br>List[int] size = [7, 4544]                   | Unknown  |
|  616 | Tensor<[1, 7, 4672]> self = ?,<br>List[int] size = [1, 7, 73, 64]              | Unknown  |
|  617 | Tensor<[1, 7, 7, 1024]> self = ?,<br>List[int] size = [1, 1, 7, 1, 7, 1024]    | Fallback |
|  618 | Tensor<[1, 7, 7, 1024]> self = ?,<br>List[int] size = [49, 1024]               | Fallback |
|  619 | Tensor<[1, 7, 7, 1536]> self = ?,<br>List[int] size = [49, 1536]               | Fallback |
|  620 | Tensor<[1, 7, 7, 2048]> self = ?,<br>List[int] size = [49, 2048]               | Fallback |
|  621 | Tensor<[1, 7, 7, 3072]> self = ?,<br>List[int] size = [49, 3072]               | Fallback |
|  622 | Tensor<[1, 7, 7, 4096]> self = ?,<br>List[int] size = [49, 4096]               | Fallback |
|  623 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] size = [1, 1, 7, 1, 7, 768]      | Fallback |
|  624 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] size = [49, 768]                 | Fallback |
|  625 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [-1, 7, 768]                 | Unknown  |
|  626 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [-1, 768]                    | Unknown  |
|  627 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [1, 7, 12, 64]               | Unknown  |
|  628 | Tensor<[1, 7, 768]> self = ?,<br>List[int] size = [7, 768]                     | Unknown  |
|  629 | Tensor<[1, 71, 64, 7]> self = ?,<br>List[int] size = [71, 64, 7]               | Unknown  |
|  630 | Tensor<[1, 71, 7, 64]> self = ?,<br>List[int] size = [1, 71, 7, 64]            | Unknown  |
|  631 | Tensor<[1, 71, 7, 64]> self = ?,<br>List[int] size = [71, 7, 64]               | Unknown  |
|  632 | Tensor<[1, 71, 7, 7]> self = ?,<br>List[int] size = [71, 7, 7]                 | Unknown  |
|  633 | Tensor<[1, 7392, 1, 1]> self = ?,<br>List[int] size = [1, 7392]                | Fallback |
|  634 | Tensor<[1, 768, 1, 1]> self = ?,<br>List[int] size = [1, 768]                  | Fallback |
|  635 | Tensor<[1, 768, 12, 16]> self = ?,<br>List[int] size = [1, 768, 192]           | Unknown  |
|  636 | Tensor<[1, 768, 14, 14]> self = ?,<br>List[int] size = [1, 768, 196]           | Fallback |
|  637 | Tensor<[1, 768, 144]> self = ?,<br>List[int] size = [1, 768, 12, 12]           | Fallback |
|  638 | Tensor<[1, 768, 196]> self = ?,<br>List[int] size = [768, 196]                 | Fallback |
|  639 | Tensor<[1, 768, 384]> self = ?,<br>List[int] size = [768, 384]                 | Done     |
|  640 | Tensor<[1, 768, 7, 7]> self = ?,<br>List[int] size = [1, 768, 49]              | Fallback |
|  641 | Tensor<[1, 768]> self = ?,<br>List[int] size = [1, 1, 768]                     | Done     |
|  642 | Tensor<[1, 784, 1, 1]> self = ?,<br>List[int] size = [1, 784]                  | Fallback |
|  643 | Tensor<[1, 7]> self = ?,<br>List[int] size = [-1, 7]                           | Unknown  |
|  644 | Tensor<[1, 7]> self = ?,<br>List[int] size = [1, -1]                           | Unknown  |
|  645 | Tensor<[1, 8, 1, 10]> self = ?,<br>List[int] size = [8, 1, 10]                 | Unknown  |
|  646 | Tensor<[1, 8, 1, 1]> self = ?,<br>List[int] size = [8, 1, 1]                   | Unknown  |
|  647 | Tensor<[1, 8, 1, 2]> self = ?,<br>List[int] size = [8, 1, 2]                   | Unknown  |
|  648 | Tensor<[1, 8, 1, 64]> self = ?,<br>List[int] size = [8, 1, 64]                 | Unknown  |
|  649 | Tensor<[1, 8, 1, 920]> self = ?,<br>List[int] size = [8, 1, 920]               | Unknown  |
|  650 | Tensor<[1, 8, 1, s0 + 1]> self = ?,<br>List[int] size = [8, 1, <s0 + 1>]       | Unknown  |
|  651 | Tensor<[1, 8, 10, 10]> self = ?,<br>List[int] size = [8, 10, 10]               | Done     |
|  652 | Tensor<[1, 8, 10, 64]> self = ?,<br>List[int] size = [8, 10, 64]               | Done     |
|  653 | Tensor<[1, 8, 2, 64]> self = ?,<br>List[int] size = [8, 2, 64]                 | Unknown  |
|  654 | Tensor<[1, 8, 2048, 160]> self = ?,<br>List[int] size = [8, 2048, 160]         | Unknown  |
|  655 | Tensor<[1, 8, 2048, 256]> self = ?,<br>List[int] size = [8, 2048, 256]         | Unknown  |
|  656 | Tensor<[1, 8, 2048, 32]> self = ?,<br>List[int] size = [8, 2048, 32]           | Unknown  |
|  657 | Tensor<[1, 8, 256, 160]> self = ?,<br>List[int] size = [8, 256, 160]           | Unknown  |
|  658 | Tensor<[1, 8, 256, 2048]> self = ?,<br>List[int] size = [8, 256, 2048]         | Unknown  |
|  659 | Tensor<[1, 8, 256, 256]> self = ?,<br>List[int] size = [8, 256, 256]           | Done     |
|  660 | Tensor<[1, 8, 256, 32]> self = ?,<br>List[int] size = [8, 256, 32]             | Done     |
|  661 | Tensor<[1, 8, 256, 96]> self = ?,<br>List[int] size = [8, 256, 96]             | Unknown  |
|  662 | Tensor<[1, 8, 300, 300]> self = ?,<br>List[int] size = [8, 300, 300]           | Fallback |
|  663 | Tensor<[1, 8, 300, 64]> self = ?,<br>List[int] size = [8, 300, 64]             | Done     |
|  664 | Tensor<[1, 8, 32, 2048]> self = ?,<br>List[int] size = [8, 32, 2048]           | Unknown  |
|  665 | Tensor<[1, 8, 32, 256]> self = ?,<br>List[int] size = [8, 32, 256]             | Done     |
|  666 | Tensor<[1, 8, 64, 10]> self = ?,<br>List[int] size = [8, 64, 10]               | Done     |
|  667 | Tensor<[1, 8, 64, 1]> self = ?,<br>List[int] size = [8, 64, 1]                 | Unknown  |
|  668 | Tensor<[1, 8, 64, 2]> self = ?,<br>List[int] size = [8, 64, 2]                 | Unknown  |
|  669 | Tensor<[1, 8, 64, 300]> self = ?,<br>List[int] size = [8, 64, 300]             | Fallback |
|  670 | Tensor<[1, 8, 64, s0 + 1]> self = ?,<br>List[int] size = [8, 64, <s0 + 1>]     | Unknown  |
|  671 | Tensor<[1, 8, 8, 1024]> self = ?,<br>List[int] size = [1, 1, 8, 1, 8, 1024]    | Fallback |
|  672 | Tensor<[1, 8, 8, 1024]> self = ?,<br>List[int] size = [64, 1024]               | Fallback |
|  673 | Tensor<[1, 8, 8, 1280]> self = ?,<br>List[int] size = [1, 64, 1280]            | Fallback |
|  674 | Tensor<[1, 8, 8, 1536]> self = ?,<br>List[int] size = [64, 1536]               | Fallback |
|  675 | Tensor<[1, 8, 8, 2048]> self = ?,<br>List[int] size = [64, 2048]               | Fallback |
|  676 | Tensor<[1, 8, 8, 3072]> self = ?,<br>List[int] size = [64, 3072]               | Fallback |
|  677 | Tensor<[1, 8, 8, 4096]> self = ?,<br>List[int] size = [64, 4096]               | Fallback |
|  678 | Tensor<[1, 8, 8, 768]> self = ?,<br>List[int] size = [1, 1, 8, 1, 8, 768]      | Fallback |
|  679 | Tensor<[1, 8, 8, 768]> self = ?,<br>List[int] size = [64, 768]                 | Fallback |
|  680 | Tensor<[1, 8, s0 + 1, 64]> self = ?,<br>List[int] size = [8, <s0 + 1>, 64]     | Unknown  |
|  681 | Tensor<[1, 819, 100, 136]> self = ?,<br>List[int] size = [1, -1, 91, 100, 136] | Unknown  |
|  682 | Tensor<[1, 819, 13, 17]> self = ?,<br>List[int] size = [1, -1, 91, 13, 17]     | Unknown  |
|  683 | Tensor<[1, 819, 25, 34]> self = ?,<br>List[int] size = [1, -1, 91, 25, 34]     | Unknown  |
|  684 | Tensor<[1, 819, 50, 68]> self = ?,<br>List[int] size = [1, -1, 91, 50, 68]     | Unknown  |
|  685 | Tensor<[1, 819, 7, 9]> self = ?,<br>List[int] size = [1, -1, 91, 7, 9]         | Unknown  |
|  686 | Tensor<[1, 888, 1, 1]> self = ?,<br>List[int] size = [1, 888]                  | Fallback |
|  687 | Tensor<[1, 8]> self = ?,<br>List[int] size = [-1, 2]                           | Fallback |
|  688 | Tensor<[1, 9, 1024]> self = ?,<br>List[int] size = [1, 9, 16, 64]              | Fallback |
|  689 | Tensor<[1, 9, 1024]> self = ?,<br>List[int] size = [9, 1024]                   | Done     |
|  690 | Tensor<[1, 9, 1280]> self = ?,<br>List[int] size = [1, -1, 8, 160]             | Fallback |
|  691 | Tensor<[1, 9, 128]> self = ?,<br>List[int] size = [9, 128]                     | Done     |
|  692 | Tensor<[1, 9, 16384]> self = ?,<br>List[int] size = [9, 16384]                 | Done     |
|  693 | Tensor<[1, 9, 2048]> self = ?,<br>List[int] size = [1, 9, 16, 128]             | Fallback |
|  694 | Tensor<[1, 9, 2048]> self = ?,<br>List[int] size = [9, 2048]                   | Done     |
|  695 | Tensor<[1, 9, 3072]> self = ?,<br>List[int] size = [9, 3072]                   | Done     |
|  696 | Tensor<[1, 9, 320]> self = ?,<br>List[int] size = [1, -1, 8, 40]               | Fallback |
|  697 | Tensor<[1, 9, 4096]> self = ?,<br>List[int] size = [1, 9, 64, 64]              | Fallback |
|  698 | Tensor<[1, 9, 4096]> self = ?,<br>List[int] size = [9, 4096]                   | Done     |
|  699 | Tensor<[1, 9, 640]> self = ?,<br>List[int] size = [1, -1, 8, 80]               | Fallback |
|  700 | Tensor<[1, 9, 768]> self = ?,<br>List[int] size = [1, 9, 12, 64]               | Fallback |
|  701 | Tensor<[1, 9, 768]> self = ?,<br>List[int] size = [9, 768]                     | Done     |
|  702 | Tensor<[1, 9, 8192]> self = ?,<br>List[int] size = [9, 8192]                   | Done     |
|  703 | Tensor<[1, 912, 1, 1]> self = ?,<br>List[int] size = [1, 912]                  | Fallback |
|  704 | Tensor<[1, 920]> self = ?,<br>List[int] size = [1, 1, 1, 920]                  | Unknown  |
|  705 | Tensor<[1, 9216]> self = ?,<br>List[int] size = [1, 64, 12, 12]                | Fallback |
|  706 | Tensor<[1, 960, 1, 1]> self = ?,<br>List[int] size = [1, 960]                  | Fallback |
|  707 | Tensor<[1, s0, 1280]> self = ?,<br>List[int] size = [<s0>, 1280]               | Unknown  |
|  708 | Tensor<[1, s0, 256]> self = ?,<br>List[int] size = [<s0>, 256]                 | Unknown  |
|  709 | Tensor<[1, s0, 80]> self = ?,<br>List[int] size = [<s0>, 80]                   | Unknown  |
|  710 | Tensor<[1, s10 + 1]> self = ?,<br>List[int] size = [1, -1]                     | Unknown  |
|  711 | Tensor<[10, 1024]> self = ?,<br>List[int] size = [1, 10, 1024]                 | Done     |
|  712 | Tensor<[10, 2048]> self = ?,<br>List[int] size = [1, 10, 2048]                 | Done     |
|  713 | Tensor<[10, 250002]> self = ?,<br>List[int] size = [1, 10, 250002]             | Fallback |
|  714 | Tensor<[10, 3072]> self = ?,<br>List[int] size = [1, 10, 3072]                 | Done     |
|  715 | Tensor<[10, 4096]> self = ?,<br>List[int] size = [1, 10, 4096]                 | Done     |
|  716 | Tensor<[10, 512]> self = ?,<br>List[int] size = [1, 10, 512]                   | Done     |
|  717 | Tensor<[10, 768]> self = ?,<br>List[int] size = [1, 10, 768]                   | Done     |
|  718 | Tensor<[100, 1, 2048]> self = ?,<br>List[int] size = [100, 2048]               | Unknown  |
|  719 | Tensor<[100, 1, 256]> self = ?,<br>List[int] size = [100, 256]                 | Unknown  |
|  720 | Tensor<[100, 1, 256]> self = ?,<br>List[int] size = [100, 8, 32]               | Unknown  |
|  721 | Tensor<[100, 12]> self = ?,<br>List[int] size = [-1, 2]                        | Fallback |
|  722 | Tensor<[100, 192]> self = ?,<br>List[int] size = [1, 100, 192]                 | Done     |
|  723 | Tensor<[100, 2048]> self = ?,<br>List[int] size = [100, 1, 2048]               | Unknown  |
|  724 | Tensor<[100, 256]> self = ?,<br>List[int] size = [100, 1, 256]                 | Unknown  |
|  725 | Tensor<[100, 4]> self = ?,<br>List[int] size = [1, 100, 4]                     | Done     |
|  726 | Tensor<[100, 8, 32]> self = ?,<br>List[int] size = [100, 256]                  | Unknown  |
|  727 | Tensor<[100, 92]> self = ?,<br>List[int] size = [1, 100, 92]                   | Fallback |
|  728 | Tensor<[100]> self = ?,<br>List[int] size = [-1, 1]                            | Unknown  |
|  729 | Tensor<[1024, 1024]> self = ?,<br>List[int] size = [1, 32, 32, 1024]           | Fallback |
|  730 | Tensor<[1024, 160]> self = ?,<br>List[int] size = [1, 1024, 160]               | Done     |
|  731 | Tensor<[1024, 192]> self = ?,<br>List[int] size = [1, 32, 32, 192]             | Fallback |
|  732 | Tensor<[1024, 192]> self = ?,<br>List[int] size = [16, 64, 192]                | Fallback |
|  733 | Tensor<[1024, 256]> self = ?,<br>List[int] size = [1, 1024, 256]               | Done     |
|  734 | Tensor<[1024, 256]> self = ?,<br>List[int] size = [1, 32, 32, 256]             | Fallback |
|  735 | Tensor<[1024, 256]> self = ?,<br>List[int] size = [16, 64, 256]                | Fallback |
|  736 | Tensor<[1024, 5120]> self = ?,<br>List[int] size = [1, 1024, 5120]             | Done     |
|  737 | Tensor<[1024, 576]> self = ?,<br>List[int] size = [16, 64, 576]                | Fallback |
|  738 | Tensor<[1024, 640]> self = ?,<br>List[int] size = [1, 1024, 640]               | Done     |
|  739 | Tensor<[1024, 768]> self = ?,<br>List[int] size = [1, 32, 32, 768]             | Fallback |
|  740 | Tensor<[1024, 768]> self = ?,<br>List[int] size = [16, 64, 768]                | Fallback |
|  741 | Tensor<[1024]> self = ?,<br>List[int] size = [1, -1, 1, 1]                     | Unknown  |
|  742 | Tensor<[10]> self = ?,<br>List[int] size = [-1, 1]                             | Fallback |
|  743 | Tensor<[10]> self = ?,<br>List[int] size = [1, -1]                             | Fallback |
|  744 | Tensor<[12, 1, 10]> self = ?,<br>List[int] size = [1, 12, 1, 10]               | Unknown  |
|  745 | Tensor<[12, 1, 1]> self = ?,<br>List[int] size = [1, 12, 1, 1]                 | Unknown  |
|  746 | Tensor<[12, 1, 24]> self = ?,<br>List[int] size = [1, 12, 1, 24]               | Unknown  |
|  747 | Tensor<[12, 1, 2]> self = ?,<br>List[int] size = [1, 12, 1, 2]                 | Unknown  |
|  748 | Tensor<[12, 1, 46]> self = ?,<br>List[int] size = [1, 12, 1, 46]               | Unknown  |
|  749 | Tensor<[12, 1, 64]> self = ?,<br>List[int] size = [1, 12, 1, 64]               | Unknown  |
|  750 | Tensor<[12, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 12, 1, <s0 + 1>]     | Unknown  |
|  751 | Tensor<[12, 1, s10 + 1]> self = ?,<br>List[int] size = [1, 12, 1, <s10 + 1>]   | Unknown  |
|  752 | Tensor<[12, 10, 10]> self = ?,<br>List[int] size = [1, 12, 10, 10]             | Done     |
|  753 | Tensor<[12, 10, 64]> self = ?,<br>List[int] size = [1, 12, 10, 64]             | Done     |
|  754 | Tensor<[12, 12, 12]> self = ?,<br>List[int] size = [1, 12, 12, 12]             | Done     |
|  755 | Tensor<[12, 12, 64]> self = ?,<br>List[int] size = [1, 12, 12, 64]             | Done     |
|  756 | Tensor<[12, 14, 14]> self = ?,<br>List[int] size = [1, 12, 14, 14]             | Done     |
|  757 | Tensor<[12, 14, 64]> self = ?,<br>List[int] size = [1, 12, 14, 64]             | Done     |
|  758 | Tensor<[12, 16, 16]> self = ?,<br>List[int] size = [1, 12, 16, 16]             | Done     |
|  759 | Tensor<[12, 16, 64]> self = ?,<br>List[int] size = [1, 12, 16, 64]             | Done     |
|  760 | Tensor<[12, 197, 197]> self = ?,<br>List[int] size = [1, 12, 197, 197]         | Fallback |
|  761 | Tensor<[12, 197, 64]> self = ?,<br>List[int] size = [1, 12, 197, 64]           | Done     |
|  762 | Tensor<[12, 201, 201]> self = ?,<br>List[int] size = [1, 12, 201, 201]         | Unknown  |
|  763 | Tensor<[12, 201, 64]> self = ?,<br>List[int] size = [1, 12, 201, 64]           | Unknown  |
|  764 | Tensor<[12, 24, 24]> self = ?,<br>List[int] size = [1, 12, 24, 24]             | Done     |
|  765 | Tensor<[12, 24, 24]> self = ?,<br>List[int] size = [12, 24, 24]                | Done     |
|  766 | Tensor<[12, 24, 64]> self = ?,<br>List[int] size = [1, 12, 24, 64]             | Done     |
|  767 | Tensor<[12, 24, 64]> self = ?,<br>List[int] size = [12, -1, 64]                | Fallback |
|  768 | Tensor<[12, 25, 25]> self = ?,<br>List[int] size = [1, 12, 25, 25]             | Done     |
|  769 | Tensor<[12, 25, 64]> self = ?,<br>List[int] size = [1, 12, 25, 64]             | Done     |
|  770 | Tensor<[12, 2]> self = ?,<br>List[int] size = [1, 12, 2]                       | Done     |
|  771 | Tensor<[12, 3072]> self = ?,<br>List[int] size = [1, 12, 3072]                 | Done     |
|  772 | Tensor<[12, 45, 45]> self = ?,<br>List[int] size = [1, 12, 45, 45]             | Unknown  |
|  773 | Tensor<[12, 45, 64]> self = ?,<br>List[int] size = [1, 12, 45, 64]             | Unknown  |
|  774 | Tensor<[12, 50, 64]> self = ?,<br>List[int] size = [1, 12, 50, 64]             | Done     |
|  775 | Tensor<[12, 7, 64]> self = ?,<br>List[int] size = [1, 12, 7, 64]               | Unknown  |
|  776 | Tensor<[12, 7, 7]> self = ?,<br>List[int] size = [1, 12, 7, 7]                 | Unknown  |
|  777 | Tensor<[12, 768]> self = ?,<br>List[int] size = [1, 12, 768]                   | Done     |
|  778 | Tensor<[12, 9, 64]> self = ?,<br>List[int] size = [1, 12, 9, 64]               | Done     |
|  779 | Tensor<[12, 9, 9]> self = ?,<br>List[int] size = [1, 12, 9, 9]                 | Done     |
|  780 | Tensor<[1200, 1280]> self = ?,<br>List[int] size = [1, 1200, 1280]             | Done     |
|  781 | Tensor<[1200, 320]> self = ?,<br>List[int] size = [1, 1200, 320]               | Done     |
|  782 | Tensor<[128, 49, 32]> self = ?,<br>List[int] size = [16, 8, 49, 32]            | Fallback |
|  783 | Tensor<[128, 49, 49]> self = ?,<br>List[int] size = [16, 8, 49, 49]            | Fallback |
|  784 | Tensor<[128, 64, 32]> self = ?,<br>List[int] size = [16, 8, 64, 32]            | Fallback |
|  785 | Tensor<[128, 64, 64]> self = ?,<br>List[int] size = [16, 8, 64, 64]            | Fallback |
|  786 | Tensor<[128]> self = ?,<br>List[int] size = [1, -1, 1, 1]                      | Unknown  |
|  787 | Tensor<[12]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  |
|  788 | Tensor<[13600, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                    | Unknown  |
|  789 | Tensor<[13600, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                    | Unknown  |
|  790 | Tensor<[136]> self = ?,<br>List[int] size = [1, -1]                            | Unknown  |
|  791 | Tensor<[1370, 1, 1280]> self = ?,<br>List[int] size = [1370, 1280]             | Fallback |
|  792 | Tensor<[1370, 1, 1280]> self = ?,<br>List[int] size = [1370, 16, 80]           | Fallback |
|  793 | Tensor<[1370, 1, 16, 80]> self = ?,<br>List[int] size = [1370, 1280]           | Fallback |
|  794 | Tensor<[1370, 1, 3840]> self = ?,<br>List[int] size = [1370, 1, 3, 1280]       | Fallback |
|  795 | Tensor<[1370, 1280]> self = ?,<br>List[int] size = [1, 1370, 1280]             | Done     |
|  796 | Tensor<[1370, 1280]> self = ?,<br>List[int] size = [1370, 1, 1280]             | Fallback |
|  797 | Tensor<[1370, 3840]> self = ?,<br>List[int] size = [1370, 1, 3840]             | Fallback |
|  798 | Tensor<[1370, 5120]> self = ?,<br>List[int] size = [1, 1370, 5120]             | Done     |
|  799 | Tensor<[13]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  |
|  800 | Tensor<[14, 14]> self = ?,<br>List[int] size = [2, 7, 2, 7]                    | Fallback |
|  801 | Tensor<[14, 2048]> self = ?,<br>List[int] size = [2, 7, 2048]                  | Fallback |
|  802 | Tensor<[14, 2]> self = ?,<br>List[int] size = [1, 14, 2]                       | Done     |
|  803 | Tensor<[14, 3072]> self = ?,<br>List[int] size = [1, 14, 3072]                 | Done     |
|  804 | Tensor<[14, 512]> self = ?,<br>List[int] size = [2, 7, 512]                    | Fallback |
|  805 | Tensor<[14, 768]> self = ?,<br>List[int] size = [1, 14, 768]                   | Done     |
|  806 | Tensor<[1444, 8]> self = ?,<br>List[int] size = [-1, 2]                        | Fallback |
|  807 | Tensor<[1445, 192]> self = ?,<br>List[int] size = [1, 1445, 192]               | Done     |
|  808 | Tensor<[1445, 768]> self = ?,<br>List[int] size = [1, 1445, 768]               | Done     |
|  809 | Tensor<[15, 1024]> self = ?,<br>List[int] size = [1, 15, 1024]                 | Done     |
|  810 | Tensor<[15, 384]> self = ?,<br>List[int] size = [1, 15, 384]                   | Done     |
|  811 | Tensor<[15, 512]> self = ?,<br>List[int] size = [1, 15, 512]                   | Done     |
|  812 | Tensor<[1500, 3072]> self = ?,<br>List[int] size = [1, 1500, 3072]             | Done     |
|  813 | Tensor<[1500, 768]> self = ?,<br>List[int] size = [1, 1500, 768]               | Done     |
|  814 | Tensor<[16, 1, 10]> self = ?,<br>List[int] size = [1, 16, 1, 10]               | Unknown  |
|  815 | Tensor<[16, 1, 1]> self = ?,<br>List[int] size = [1, 16, 1, 1]                 | Unknown  |
|  816 | Tensor<[16, 1, 2]> self = ?,<br>List[int] size = [1, 16, 1, 2]                 | Unknown  |
|  817 | Tensor<[16, 1, 60]> self = ?,<br>List[int] size = [1, 16, 1, 60]               | Unknown  |
|  818 | Tensor<[16, 1, 64]> self = ?,<br>List[int] size = [1, 16, 1, 64]               | Unknown  |
|  819 | Tensor<[16, 1, 6]> self = ?,<br>List[int] size = [1, 16, 1, 6]                 | Unknown  |
|  820 | Tensor<[16, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 16, 1, <s0 + 1>]     | Unknown  |
|  821 | Tensor<[16, 1, s10 + 1]> self = ?,<br>List[int] size = [1, 16, 1, <s10 + 1>]   | Unknown  |
|  822 | Tensor<[16, 10, 10]> self = ?,<br>List[int] size = [1, 16, 10, 10]             | Done     |
|  823 | Tensor<[16, 10, 64]> self = ?,<br>List[int] size = [1, 16, 10, 64]             | Done     |
|  824 | Tensor<[16, 1370, 80]> self = ?,<br>List[int] size = [1, 16, 1370, 80]         | Fallback |
|  825 | Tensor<[16, 16]> self = ?,<br>List[int] size = [2, 8, 2, 8]                    | Fallback |
|  826 | Tensor<[16, 19, 19]> self = ?,<br>List[int] size = [1, 16, 19, 19]             | Done     |
|  827 | Tensor<[16, 19, 64]> self = ?,<br>List[int] size = [1, 16, 19, 64]             | Done     |
|  828 | Tensor<[16, 197, 197]> self = ?,<br>List[int] size = [1, 16, 197, 197]         | Fallback |
|  829 | Tensor<[16, 197, 64]> self = ?,<br>List[int] size = [1, 16, 197, 64]           | Done     |
|  830 | Tensor<[16, 256, 256]> self = ?,<br>List[int] size = [1, 16, 256, 256]         | Done     |
|  831 | Tensor<[16, 256, 64]> self = ?,<br>List[int] size = [1, 16, 256, 64]           | Done     |
|  832 | Tensor<[16, 3072]> self = ?,<br>List[int] size = [1, 16, 3072]                 | Done     |
|  833 | Tensor<[16, 32, 32]> self = ?,<br>List[int] size = [1, 16, 32, 32]             | Done     |
|  834 | Tensor<[16, 32, 96]> self = ?,<br>List[int] size = [1, 16, 32, 96]             | Done     |
|  835 | Tensor<[16, 49, 192]> self = ?,<br>List[int] size = [1, 4, 4, 7, 7, 192]       | Fallback |
|  836 | Tensor<[16, 49, 192]> self = ?,<br>List[int] size = [784, 192]                 | Fallback |
|  837 | Tensor<[16, 49, 256]> self = ?,<br>List[int] size = [1, 4, 4, 7, 7, 256]       | Fallback |
|  838 | Tensor<[16, 49, 256]> self = ?,<br>List[int] size = [784, 256]                 | Fallback |
|  839 | Tensor<[16, 49, 576]> self = ?,<br>List[int] size = [16, 49, 3, 6, 32]         | Fallback |
|  840 | Tensor<[16, 49, 768]> self = ?,<br>List[int] size = [16, 49, 3, 8, 32]         | Fallback |
|  841 | Tensor<[16, 5, 5]> self = ?,<br>List[int] size = [1, 16, 5, 5]                 | Unknown  |
|  842 | Tensor<[16, 5, 64]> self = ?,<br>List[int] size = [1, 16, 5, 64]               | Unknown  |
|  843 | Tensor<[16, 50, 64]> self = ?,<br>List[int] size = [1, 16, 50, 64]             | Done     |
|  844 | Tensor<[16, 59, 59]> self = ?,<br>List[int] size = [1, 16, 59, 59]             | Fallback |
|  845 | Tensor<[16, 59, 64]> self = ?,<br>List[int] size = [1, 16, 59, 64]             | Done     |
|  846 | Tensor<[16, 6, 49, 49]> self = ?,<br>List[int] size = [1, 16, 6, 49, 49]       | Fallback |
|  847 | Tensor<[16, 6, 49, 49]> self = ?,<br>List[int] size = [96, 49, 49]             | Fallback |
|  848 | Tensor<[16, 6, 64, 64]> self = ?,<br>List[int] size = [1, 16, 6, 64, 64]       | Fallback |
|  849 | Tensor<[16, 6, 64, 64]> self = ?,<br>List[int] size = [96, 64, 64]             | Fallback |
|  850 | Tensor<[16, 64, 192]> self = ?,<br>List[int] size = [1, 4, 4, 8, 8, 192]       | Fallback |
|  851 | Tensor<[16, 64, 192]> self = ?,<br>List[int] size = [1024, 192]                | Fallback |
|  852 | Tensor<[16, 64, 256]> self = ?,<br>List[int] size = [1, 4, 4, 8, 8, 256]       | Fallback |
|  853 | Tensor<[16, 64, 256]> self = ?,<br>List[int] size = [1024, 256]                | Fallback |
|  854 | Tensor<[16, 64, 576]> self = ?,<br>List[int] size = [16, 64, 3, 6, 32]         | Fallback |
|  855 | Tensor<[16, 64, 768]> self = ?,<br>List[int] size = [16, 64, 3, 8, 32]         | Fallback |
|  856 | Tensor<[16, 7, 64]> self = ?,<br>List[int] size = [2, 8, 7, 64]                | Fallback |
|  857 | Tensor<[16, 7, 7]> self = ?,<br>List[int] size = [2, 8, 7, 7]                  | Fallback |
|  858 | Tensor<[16, 768]> self = ?,<br>List[int] size = [1, 16, 768]                   | Done     |
|  859 | Tensor<[16, 8, 49, 49]> self = ?,<br>List[int] size = [1, 16, 8, 49, 49]       | Fallback |
|  860 | Tensor<[16, 8, 49, 49]> self = ?,<br>List[int] size = [128, 49, 49]            | Fallback |
|  861 | Tensor<[16, 8, 64, 64]> self = ?,<br>List[int] size = [1, 16, 8, 64, 64]       | Fallback |
|  862 | Tensor<[16, 8, 64, 64]> self = ?,<br>List[int] size = [128, 64, 64]            | Fallback |
|  863 | Tensor<[16, 9, 128]> self = ?,<br>List[int] size = [1, 16, 9, 128]             | Done     |
|  864 | Tensor<[16, 9, 64]> self = ?,<br>List[int] size = [1, 16, 9, 64]               | Done     |
|  865 | Tensor<[16, 9, 9]> self = ?,<br>List[int] size = [1, 16, 9, 9]                 | Done     |
|  866 | Tensor<[16384, 128]> self = ?,<br>List[int] size = [1, 16384, 128]             | Done     |
|  867 | Tensor<[16384, 256]> self = ?,<br>List[int] size = [1, 16384, 256]             | Done     |
|  868 | Tensor<[16384, 32]> self = ?,<br>List[int] size = [1, 16384, 32]               | Done     |
|  869 | Tensor<[16]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  |
|  870 | Tensor<[17]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  |
|  871 | Tensor<[19, 1024]> self = ?,<br>List[int] size = [1, 19, 1024]                 | Done     |
|  872 | Tensor<[19, 256008]> self = ?,<br>List[int] size = [1, 19, 256008]             | Fallback |
|  873 | Tensor<[19, 4096]> self = ?,<br>List[int] size = [1, 19, 4096]                 | Done     |
|  874 | Tensor<[192, 49, 32]> self = ?,<br>List[int] size = [64, 3, 49, 32]            | Fallback |
|  875 | Tensor<[192, 49, 49]> self = ?,<br>List[int] size = [64, 3, 49, 49]            | Fallback |
|  876 | Tensor<[192, 64, 32]> self = ?,<br>List[int] size = [64, 3, 64, 32]            | Fallback |
|  877 | Tensor<[192, 64, 64]> self = ?,<br>List[int] size = [64, 3, 64, 64]            | Fallback |
|  878 | Tensor<[19200, 256]> self = ?,<br>List[int] size = [1, 19200, 256]             | Done     |
|  879 | Tensor<[19200, 64]> self = ?,<br>List[int] size = [1, 19200, 64]               | Done     |
|  880 | Tensor<[192]> self = ?,<br>List[int] size = [1, 192, 1, 1]                     | Fallback |
|  881 | Tensor<[196, 1152]> self = ?,<br>List[int] size = [4, 49, 1152]                | Fallback |
|  882 | Tensor<[196, 1536]> self = ?,<br>List[int] size = [1, 14, 14, 1536]            | Fallback |
|  883 | Tensor<[196, 1536]> self = ?,<br>List[int] size = [4, 49, 1536]                | Fallback |
|  884 | Tensor<[196, 2048]> self = ?,<br>List[int] size = [1, 14, 14, 2048]            | Fallback |
|  885 | Tensor<[196, 3072]> self = ?,<br>List[int] size = [1, 196, 3072]               | Done     |
|  886 | Tensor<[196, 384]> self = ?,<br>List[int] size = [1, 14, 14, 384]              | Fallback |
|  887 | Tensor<[196, 384]> self = ?,<br>List[int] size = [4, 49, 384]                  | Fallback |
|  888 | Tensor<[196, 512]> self = ?,<br>List[int] size = [1, 14, 14, 512]              | Fallback |
|  889 | Tensor<[196, 512]> self = ?,<br>List[int] size = [4, 49, 512]                  | Fallback |
|  890 | Tensor<[196, 768]> self = ?,<br>List[int] size = [1, 196, 768]                 | Done     |
|  891 | Tensor<[197, 1, 1024]> self = ?,<br>List[int] size = [197, 1024]               | Fallback |
|  892 | Tensor<[197, 1, 1024]> self = ?,<br>List[int] size = [197, 16, 64]             | Fallback |
|  893 | Tensor<[197, 1, 12, 64]> self = ?,<br>List[int] size = [197, 768]              | Fallback |
|  894 | Tensor<[197, 1, 16, 64]> self = ?,<br>List[int] size = [197, 1024]             | Fallback |
|  895 | Tensor<[197, 1, 2304]> self = ?,<br>List[int] size = [197, 1, 3, 768]          | Fallback |
|  896 | Tensor<[197, 1, 3072]> self = ?,<br>List[int] size = [197, 1, 3, 1024]         | Fallback |
|  897 | Tensor<[197, 1, 768]> self = ?,<br>List[int] size = [197, 12, 64]              | Fallback |
|  898 | Tensor<[197, 1, 768]> self = ?,<br>List[int] size = [197, 768]                 | Fallback |
|  899 | Tensor<[197, 1024]> self = ?,<br>List[int] size = [1, 197, 1024]               | Done     |
|  900 | Tensor<[197, 1024]> self = ?,<br>List[int] size = [197, 1, 1024]               | Fallback |
|  901 | Tensor<[197, 197]> self = ?,<br>List[int] size = [-1]                          | Fallback |
|  902 | Tensor<[197, 2304]> self = ?,<br>List[int] size = [197, 1, 2304]               | Fallback |
|  903 | Tensor<[197, 3072]> self = ?,<br>List[int] size = [1, 197, 3072]               | Done     |
|  904 | Tensor<[197, 3072]> self = ?,<br>List[int] size = [197, 1, 3072]               | Fallback |
|  905 | Tensor<[197, 4096]> self = ?,<br>List[int] size = [1, 197, 4096]               | Done     |
|  906 | Tensor<[197, 768]> self = ?,<br>List[int] size = [1, 197, 768]                 | Done     |
|  907 | Tensor<[197, 768]> self = ?,<br>List[int] size = [197, 1, 768]                 | Fallback |
|  908 | Tensor<[19]> self = ?,<br>List[int] size = [-1, 1]                             | Fallback |
|  909 | Tensor<[19]> self = ?,<br>List[int] size = [1, -1]                             | Fallback |
|  910 | Tensor<[1]> self = ?,<br>List[int] size = [-1, 1]                              | Fallback |
|  911 | Tensor<[1]> self = ?,<br>List[int] size = [1, -1]                              | Fallback |
|  912 | Tensor<[1]> self = ?,<br>List[int] size = [1, 1, 1, 1]                         | Fallback |
|  913 | Tensor<[2, 4096, 256]> self = ?,<br>List[int] size = [1, 2, 4096, 256]         | Done     |
|  914 | Tensor<[2, 4096, 32]> self = ?,<br>List[int] size = [1, 2, 4096, 32]           | Done     |
|  915 | Tensor<[2, 4800, 300]> self = ?,<br>List[int] size = [1, 2, 4800, 300]         | Fallback |
|  916 | Tensor<[2, 4800, 64]> self = ?,<br>List[int] size = [1, 2, 4800, 64]           | Done     |
|  917 | Tensor<[2, 7, 2048]> self = ?,<br>List[int] size = [14, 2048]                  | Fallback |
|  918 | Tensor<[2, 7, 512]> self = ?,<br>List[int] size = [14, 512]                    | Fallback |
|  919 | Tensor<[2, 7, 512]> self = ?,<br>List[int] size = [2, -1, 8, 64]               | Fallback |
|  920 | Tensor<[2, 7, 512]> self = ?,<br>List[int] size = [2, 7, 8, 64]                | Fallback |
|  921 | Tensor<[2, 7]> self = ?,<br>List[int] size = [-1, 7]                           | Fallback |
|  922 | Tensor<[2, 8, 7, 64]> self = ?,<br>List[int] size = [16, -1, 64]               | Fallback |
|  923 | Tensor<[2, 8, 7, 7]> self = ?,<br>List[int] size = [16, 7, 7]                  | Fallback |
|  924 | Tensor<[201, 3072]> self = ?,<br>List[int] size = [1, 201, 3072]               | Unknown  |
|  925 | Tensor<[201, 768]> self = ?,<br>List[int] size = [1, 201, 768]                 | Unknown  |
|  926 | Tensor<[2048, 1280]> self = ?,<br>List[int] size = [1, 2048, 1280]             | Unknown  |
|  927 | Tensor<[2048, 256]> self = ?,<br>List[int] size = [1, 2048, 256]               | Unknown  |
|  928 | Tensor<[2048, 262]> self = ?,<br>List[int] size = [1, 2048, 262]               | Unknown  |
|  929 | Tensor<[2048, 768]> self = ?,<br>List[int] size = [1, 2048, 768]               | Unknown  |
|  930 | Tensor<[2048]> self = ?,<br>List[int] size = [1, -1, 1, 1]                     | Unknown  |
|  931 | Tensor<[20]> self = ?,<br>List[int] size = [-1, 1]                             | Fallback |
|  932 | Tensor<[20]> self = ?,<br>List[int] size = [1, -1]                             | Fallback |
|  933 | Tensor<[221, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                      | Unknown  |
|  934 | Tensor<[221, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                      | Unknown  |
|  935 | Tensor<[225, 12]> self = ?,<br>List[int] size = [1, 15, 15, 12]                | Fallback |
|  936 | Tensor<[225, 16]> self = ?,<br>List[int] size = [1, 15, 15, 16]                | Fallback |
|  937 | Tensor<[225, 24]> self = ?,<br>List[int] size = [1, 15, 15, 24]                | Fallback |
|  938 | Tensor<[225, 32]> self = ?,<br>List[int] size = [1, 15, 15, 32]                | Fallback |
|  939 | Tensor<[225, 3]> self = ?,<br>List[int] size = [1, 15, 15, 3]                  | Fallback |
|  940 | Tensor<[225, 4]> self = ?,<br>List[int] size = [1, 15, 15, 4]                  | Fallback |
|  941 | Tensor<[225, 512]> self = ?,<br>List[int] size = [1, 15, 15, 512]              | Fallback |
|  942 | Tensor<[225, 6]> self = ?,<br>List[int] size = [1, 15, 15, 6]                  | Fallback |
|  943 | Tensor<[225, 8]> self = ?,<br>List[int] size = [1, 15, 15, 8]                  | Fallback |
|  944 | Tensor<[24, 12, 24]> self = ?,<br>List[int] size = [24, 12, 24]                | Done     |
|  945 | Tensor<[24, 12, 64]> self = ?,<br>List[int] size = [24, 12, 64]                | Done     |
|  946 | Tensor<[24, 3072]> self = ?,<br>List[int] size = [1, 24, 3072]                 | Done     |
|  947 | Tensor<[24, 49, 32]> self = ?,<br>List[int] size = [1, 24, 49, 32]             | Done     |
|  948 | Tensor<[24, 49, 49]> self = ?,<br>List[int] size = [1, 24, 49, 49]             | Fallback |
|  949 | Tensor<[24, 64, 24]> self = ?,<br>List[int] size = [24, 64, 24]                | Done     |
|  950 | Tensor<[24, 64, 32]> self = ?,<br>List[int] size = [1, 24, 64, 32]             | Done     |
|  951 | Tensor<[24, 64, 64]> self = ?,<br>List[int] size = [1, 24, 64, 64]             | Done     |
|  952 | Tensor<[24, 768]> self = ?,<br>List[int] size = [1, 24, 768]                   | Done     |
|  953 | Tensor<[2401, 12]> self = ?,<br>List[int] size = [49, 49, -1]                  | Fallback |
|  954 | Tensor<[2401, 16]> self = ?,<br>List[int] size = [49, 49, -1]                  | Fallback |
|  955 | Tensor<[2401, 24]> self = ?,<br>List[int] size = [49, 49, -1]                  | Fallback |
|  956 | Tensor<[2401, 32]> self = ?,<br>List[int] size = [49, 49, -1]                  | Fallback |
|  957 | Tensor<[2401, 3]> self = ?,<br>List[int] size = [49, 49, -1]                   | Fallback |
|  958 | Tensor<[2401, 4]> self = ?,<br>List[int] size = [49, 49, -1]                   | Fallback |
|  959 | Tensor<[2401, 6]> self = ?,<br>List[int] size = [49, 49, -1]                   | Fallback |
|  960 | Tensor<[2401, 8]> self = ?,<br>List[int] size = [49, 49, -1]                   | Fallback |
|  961 | Tensor<[24576, 1]> self = ?,<br>List[int] size = [-1]                          | Unknown  |
|  962 | Tensor<[25, 12]> self = ?,<br>List[int] size = [-1, 2]                         | Fallback |
|  963 | Tensor<[25, 2]> self = ?,<br>List[int] size = [1, 25, 2]                       | Done     |
|  964 | Tensor<[25, 3072]> self = ?,<br>List[int] size = [1, 25, 3072]                 | Done     |
|  965 | Tensor<[25, 768]> self = ?,<br>List[int] size = [1, 25, 768]                   | Done     |
|  966 | Tensor<[256, 10240]> self = ?,<br>List[int] size = [1, 256, 10240]             | Done     |
|  967 | Tensor<[256, 1024]> self = ?,<br>List[int] size = [1, 256, 1024]               | Done     |
|  968 | Tensor<[256, 1152]> self = ?,<br>List[int] size = [4, 64, 1152]                | Fallback |
|  969 | Tensor<[256, 1280]> self = ?,<br>List[int] size = [1, 256, 1280]               | Done     |
|  970 | Tensor<[256, 1536]> self = ?,<br>List[int] size = [1, 16, 16, 1536]            | Fallback |
|  971 | Tensor<[256, 1536]> self = ?,<br>List[int] size = [4, 64, 1536]                | Fallback |
|  972 | Tensor<[256, 160]> self = ?,<br>List[int] size = [1, 256, 160]                 | Done     |
|  973 | Tensor<[256, 2048]> self = ?,<br>List[int] size = [1, 16, 16, 2048]            | Fallback |
|  974 | Tensor<[256, 256]> self = ?,<br>List[int] size = [1, 256, 256]                 | Done     |
|  975 | Tensor<[256, 2]> self = ?,<br>List[int] size = [1, 256, 2]                     | Done     |
|  976 | Tensor<[256, 32]> self = ?,<br>List[int] size = [1, 256, 32]                   | Done     |
|  977 | Tensor<[256, 384]> self = ?,<br>List[int] size = [1, 16, 16, 384]              | Fallback |
|  978 | Tensor<[256, 384]> self = ?,<br>List[int] size = [4, 64, 384]                  | Fallback |
|  979 | Tensor<[256, 4096]> self = ?,<br>List[int] size = [1, 256, 4096]               | Done     |
|  980 | Tensor<[256, 49, 32]> self = ?,<br>List[int] size = [64, 4, 49, 32]            | Fallback |
|  981 | Tensor<[256, 49, 49]> self = ?,<br>List[int] size = [64, 4, 49, 49]            | Fallback |
|  982 | Tensor<[256, 512]> self = ?,<br>List[int] size = [1, 16, 16, 512]              | Fallback |
|  983 | Tensor<[256, 512]> self = ?,<br>List[int] size = [1, 256, 512]                 | Done     |
|  984 | Tensor<[256, 512]> self = ?,<br>List[int] size = [4, 64, 512]                  | Fallback |
|  985 | Tensor<[256, 64, 32]> self = ?,<br>List[int] size = [64, 4, 64, 32]            | Fallback |
|  986 | Tensor<[256, 64, 64]> self = ?,<br>List[int] size = [64, 4, 64, 64]            | Fallback |
|  987 | Tensor<[256, 64]> self = ?,<br>List[int] size = [1, 256, 64]                   | Done     |
|  988 | Tensor<[256, 768]> self = ?,<br>List[int] size = [1, 256, 768]                 | Unknown  |
|  989 | Tensor<[256]> self = ?,<br>List[int] size = [1, -1, 1, 1]                      | Unknown  |
|  990 | Tensor<[25]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  |
|  991 | Tensor<[28, 28]> self = ?,<br>List[int] size = [4, 7, 4, 7]                    | Fallback |
|  992 | Tensor<[2]> self = ?,<br>List[int] size = [-1, 1]                              | Fallback |
|  993 | Tensor<[2]> self = ?,<br>List[int] size = [1, -1]                              | Fallback |
|  994 | Tensor<[3, 1445, 1445]> self = ?,<br>List[int] size = [1, 3, 1445, 1445]       | Fallback |
|  995 | Tensor<[3, 1445, 64]> self = ?,<br>List[int] size = [1, 3, 1445, 64]           | Done     |
|  996 | Tensor<[300, 128]> self = ?,<br>List[int] size = [1, 300, 128]                 | Done     |
|  997 | Tensor<[300, 2048]> self = ?,<br>List[int] size = [1, 300, 2048]               | Done     |
|  998 | Tensor<[300, 320]> self = ?,<br>List[int] size = [1, 300, 320]                 | Done     |
|  999 | Tensor<[300, 512]> self = ?,<br>List[int] size = [1, 300, 512]                 | Done     |
| 1000 | Tensor<[300, 64]> self = ?,<br>List[int] size = [1, 300, 64]                   | Done     |
| 1001 | Tensor<[3136, 128]> self = ?,<br>List[int] size = [1, 56, 56, 128]             | Fallback |
| 1002 | Tensor<[3136, 128]> self = ?,<br>List[int] size = [64, 49, 128]                | Fallback |
| 1003 | Tensor<[3136, 288]> self = ?,<br>List[int] size = [64, 49, 288]                | Fallback |
| 1004 | Tensor<[3136, 384]> self = ?,<br>List[int] size = [1, 56, 56, 384]             | Fallback |
| 1005 | Tensor<[3136, 384]> self = ?,<br>List[int] size = [64, 49, 384]                | Fallback |
| 1006 | Tensor<[3136, 512]> self = ?,<br>List[int] size = [1, 56, 56, 512]             | Fallback |
| 1007 | Tensor<[3136, 96]> self = ?,<br>List[int] size = [1, 56, 56, 96]               | Fallback |
| 1008 | Tensor<[3136, 96]> self = ?,<br>List[int] size = [64, 49, 96]                  | Fallback |
| 1009 | Tensor<[32, 1536]> self = ?,<br>List[int] size = [1, 32, 1536]                 | Done     |
| 1010 | Tensor<[32, 250880]> self = ?,<br>List[int] size = [1, 32, 250880]             | Done     |
| 1011 | Tensor<[32, 32]> self = ?,<br>List[int] size = [4, 8, 4, 8]                    | Fallback |
| 1012 | Tensor<[32, 4608]> self = ?,<br>List[int] size = [1, 32, 4608]                 | Done     |
| 1013 | Tensor<[32, 49, 32]> self = ?,<br>List[int] size = [1, 32, 49, 32]             | Done     |
| 1014 | Tensor<[32, 49, 49]> self = ?,<br>List[int] size = [1, 32, 49, 49]             | Fallback |
| 1015 | Tensor<[32, 6144]> self = ?,<br>List[int] size = [1, 32, 6144]                 | Done     |
| 1016 | Tensor<[32, 64, 32]> self = ?,<br>List[int] size = [1, 32, 64, 32]             | Done     |
| 1017 | Tensor<[32, 64, 64]> self = ?,<br>List[int] size = [1, 32, 64, 64]             | Done     |
| 1018 | Tensor<[3234, 1, 4]> self = ?,<br>List[int] size = [3234, 4]                   | Unknown  |
| 1019 | Tensor<[3234, 2, 2]> self = ?,<br>List[int] size = [3234, 4]                   | Unknown  |
| 1020 | Tensor<[32]> self = ?,<br>List[int] size = [1, 1, 32, 1]                       | Fallback |
| 1021 | Tensor<[3400, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                     | Unknown  |
| 1022 | Tensor<[3400, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                     | Unknown  |
| 1023 | Tensor<[34]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  |
| 1024 | Tensor<[361, 12]> self = ?,<br>List[int] size = [-1, 2]                        | Fallback |
| 1025 | Tensor<[38809, 12]> self = ?,<br>List[int] size = [197, 197, -1]               | Fallback |
| 1026 | Tensor<[38809, 16]> self = ?,<br>List[int] size = [197, 197, -1]               | Fallback |
| 1027 | Tensor<[38]> self = ?,<br>List[int] size = [-1, 1]                             | Fallback |
| 1028 | Tensor<[38]> self = ?,<br>List[int] size = [1, -1]                             | Fallback |
| 1029 | Tensor<[3]> self = ?,<br>List[int] size = [-1, 1]                              | Fallback |
| 1030 | Tensor<[3]> self = ?,<br>List[int] size = [1, -1]                              | Fallback |
| 1031 | Tensor<[4, 12, 49, 49]> self = ?,<br>List[int] size = [1, 4, 12, 49, 49]       | Fallback |
| 1032 | Tensor<[4, 12, 49, 49]> self = ?,<br>List[int] size = [48, 49, 49]             | Fallback |
| 1033 | Tensor<[4, 12, 64, 64]> self = ?,<br>List[int] size = [1, 4, 12, 64, 64]       | Fallback |
| 1034 | Tensor<[4, 12, 64, 64]> self = ?,<br>List[int] size = [48, 64, 64]             | Fallback |
| 1035 | Tensor<[4, 12]> self = ?,<br>List[int] size = [-1, 2]                          | Fallback |
| 1036 | Tensor<[4, 16, 49, 49]> self = ?,<br>List[int] size = [1, 4, 16, 49, 49]       | Fallback |
| 1037 | Tensor<[4, 16, 49, 49]> self = ?,<br>List[int] size = [64, 49, 49]             | Fallback |
| 1038 | Tensor<[4, 16, 64, 64]> self = ?,<br>List[int] size = [1, 4, 16, 64, 64]       | Fallback |
| 1039 | Tensor<[4, 16, 64, 64]> self = ?,<br>List[int] size = [64, 64, 64]             | Fallback |
| 1040 | Tensor<[4, 3072]> self = ?,<br>List[int] size = [1, 4, 3072]                   | Unknown  |
| 1041 | Tensor<[4, 49, 1152]> self = ?,<br>List[int] size = [4, 49, 3, 12, 32]         | Fallback |
| 1042 | Tensor<[4, 49, 1536]> self = ?,<br>List[int] size = [4, 49, 3, 16, 32]         | Fallback |
| 1043 | Tensor<[4, 49, 384]> self = ?,<br>List[int] size = [1, 2, 2, 7, 7, 384]        | Fallback |
| 1044 | Tensor<[4, 49, 384]> self = ?,<br>List[int] size = [196, 384]                  | Fallback |
| 1045 | Tensor<[4, 49, 512]> self = ?,<br>List[int] size = [1, 2, 2, 7, 7, 512]        | Fallback |
| 1046 | Tensor<[4, 49, 512]> self = ?,<br>List[int] size = [196, 512]                  | Fallback |
| 1047 | Tensor<[4, 51865]> self = ?,<br>List[int] size = [1, 4, 51865]                 | Unknown  |
| 1048 | Tensor<[4, 64, 1152]> self = ?,<br>List[int] size = [4, 64, 3, 12, 32]         | Fallback |
| 1049 | Tensor<[4, 64, 1536]> self = ?,<br>List[int] size = [4, 64, 3, 16, 32]         | Fallback |
| 1050 | Tensor<[4, 64, 384]> self = ?,<br>List[int] size = [1, 2, 2, 8, 8, 384]        | Fallback |
| 1051 | Tensor<[4, 64, 384]> self = ?,<br>List[int] size = [256, 384]                  | Fallback |
| 1052 | Tensor<[4, 64, 512]> self = ?,<br>List[int] size = [1, 2, 2, 8, 8, 512]        | Fallback |
| 1053 | Tensor<[4, 64, 512]> self = ?,<br>List[int] size = [256, 512]                  | Fallback |
| 1054 | Tensor<[4, 768]> self = ?,<br>List[int] size = [1, 4, 768]                     | Unknown  |
| 1055 | Tensor<[400, 12]> self = ?,<br>List[int] size = [-1, 2]                        | Fallback |
| 1056 | Tensor<[4096, 128]> self = ?,<br>List[int] size = [1, 64, 64, 128]             | Fallback |
| 1057 | Tensor<[4096, 128]> self = ?,<br>List[int] size = [64, 64, 128]                | Fallback |
| 1058 | Tensor<[4096, 12]> self = ?,<br>List[int] size = [64, 64, -1]                  | Fallback |
| 1059 | Tensor<[4096, 16]> self = ?,<br>List[int] size = [64, 64, -1]                  | Fallback |
| 1060 | Tensor<[4096, 24]> self = ?,<br>List[int] size = [64, 64, -1]                  | Fallback |
| 1061 | Tensor<[4096, 2560]> self = ?,<br>List[int] size = [1, 4096, 2560]             | Done     |
| 1062 | Tensor<[4096, 256]> self = ?,<br>List[int] size = [1, 4096, 256]               | Done     |
| 1063 | Tensor<[4096, 288]> self = ?,<br>List[int] size = [64, 64, 288]                | Fallback |
| 1064 | Tensor<[4096, 320]> self = ?,<br>List[int] size = [1, 4096, 320]               | Done     |
| 1065 | Tensor<[4096, 32]> self = ?,<br>List[int] size = [64, 64, -1]                  | Fallback |
| 1066 | Tensor<[4096, 384]> self = ?,<br>List[int] size = [1, 64, 64, 384]             | Fallback |
| 1067 | Tensor<[4096, 384]> self = ?,<br>List[int] size = [64, 64, 384]                | Fallback |
| 1068 | Tensor<[4096, 3]> self = ?,<br>List[int] size = [64, 64, -1]                   | Fallback |
| 1069 | Tensor<[4096, 4]> self = ?,<br>List[int] size = [64, 64, -1]                   | Fallback |
| 1070 | Tensor<[4096, 512]> self = ?,<br>List[int] size = [1, 64, 64, 512]             | Fallback |
| 1071 | Tensor<[4096, 64]> self = ?,<br>List[int] size = [1, 4096, 64]                 | Done     |
| 1072 | Tensor<[4096, 6]> self = ?,<br>List[int] size = [64, 64, -1]                   | Fallback |
| 1073 | Tensor<[4096, 8]> self = ?,<br>List[int] size = [64, 64, -1]                   | Fallback |
| 1074 | Tensor<[4096, 96]> self = ?,<br>List[int] size = [1, 64, 64, 96]               | Fallback |
| 1075 | Tensor<[4096, 96]> self = ?,<br>List[int] size = [64, 64, 96]                  | Fallback |
| 1076 | Tensor<[42]> self = ?,<br>List[int] size = [1, 1, 1, 42]                       | Fallback |
| 1077 | Tensor<[45, 3072]> self = ?,<br>List[int] size = [1, 45, 3072]                 | Unknown  |
| 1078 | Tensor<[45, 50257]> self = ?,<br>List[int] size = [1, 45, 50257]               | Unknown  |
| 1079 | Tensor<[45, 768]> self = ?,<br>List[int] size = [1, 45, 768]                   | Unknown  |
| 1080 | Tensor<[48, 49, 32]> self = ?,<br>List[int] size = [4, 12, 49, 32]             | Fallback |
| 1081 | Tensor<[48, 49, 49]> self = ?,<br>List[int] size = [4, 12, 49, 49]             | Fallback |
| 1082 | Tensor<[48, 64, 32]> self = ?,<br>List[int] size = [4, 12, 64, 32]             | Fallback |
| 1083 | Tensor<[48, 64, 64]> self = ?,<br>List[int] size = [4, 12, 64, 64]             | Fallback |
| 1084 | Tensor<[4800, 128]> self = ?,<br>List[int] size = [1, 4800, 128]               | Done     |
| 1085 | Tensor<[4800, 512]> self = ?,<br>List[int] size = [1, 4800, 512]               | Done     |
| 1086 | Tensor<[49, 1024]> self = ?,<br>List[int] size = [1, 49, 1024]                 | Done     |
| 1087 | Tensor<[49, 1024]> self = ?,<br>List[int] size = [1, 7, 7, 1024]               | Fallback |
| 1088 | Tensor<[49, 2304]> self = ?,<br>List[int] size = [1, 49, 2304]                 | Done     |
| 1089 | Tensor<[49, 3072]> self = ?,<br>List[int] size = [1, 49, 3072]                 | Done     |
| 1090 | Tensor<[49, 3072]> self = ?,<br>List[int] size = [1, 7, 7, 3072]               | Fallback |
| 1091 | Tensor<[49, 4096]> self = ?,<br>List[int] size = [1, 7, 7, 4096]               | Fallback |
| 1092 | Tensor<[49, 768]> self = ?,<br>List[int] size = [1, 49, 768]                   | Done     |
| 1093 | Tensor<[49, 768]> self = ?,<br>List[int] size = [1, 7, 7, 768]                 | Fallback |
| 1094 | Tensor<[5, 1024, 256]> self = ?,<br>List[int] size = [1, 5, 1024, 256]         | Done     |
| 1095 | Tensor<[5, 1024, 32]> self = ?,<br>List[int] size = [1, 5, 1024, 32]           | Done     |
| 1096 | Tensor<[5, 1024]> self = ?,<br>List[int] size = [1, 5, 1024]                   | Unknown  |
| 1097 | Tensor<[5, 1200, 300]> self = ?,<br>List[int] size = [1, 5, 1200, 300]         | Fallback |
| 1098 | Tensor<[5, 1200, 64]> self = ?,<br>List[int] size = [1, 5, 1200, 64]           | Done     |
| 1099 | Tensor<[5, 3072]> self = ?,<br>List[int] size = [1, 5, 3072]                   | Unknown  |
| 1100 | Tensor<[5, 4096]> self = ?,<br>List[int] size = [1, 5, 4096]                   | Unknown  |
| 1101 | Tensor<[5, 51200]> self = ?,<br>List[int] size = [1, 5, 51200]                 | Unknown  |
| 1102 | Tensor<[50, 1, 1024]> self = ?,<br>List[int] size = [50, 1024]                 | Fallback |
| 1103 | Tensor<[50, 1, 1024]> self = ?,<br>List[int] size = [50, 16, 64]               | Fallback |
| 1104 | Tensor<[50, 1, 12, 64]> self = ?,<br>List[int] size = [50, 768]                | Fallback |
| 1105 | Tensor<[50, 1, 16, 64]> self = ?,<br>List[int] size = [50, 1024]               | Fallback |
| 1106 | Tensor<[50, 1, 2304]> self = ?,<br>List[int] size = [50, 1, 3, 768]            | Fallback |
| 1107 | Tensor<[50, 1, 3072]> self = ?,<br>List[int] size = [50, 1, 3, 1024]           | Fallback |
| 1108 | Tensor<[50, 1, 768]> self = ?,<br>List[int] size = [50, 12, 64]                | Fallback |
| 1109 | Tensor<[50, 1, 768]> self = ?,<br>List[int] size = [50, 768]                   | Fallback |
| 1110 | Tensor<[50, 1024]> self = ?,<br>List[int] size = [1, 50, 1024]                 | Done     |
| 1111 | Tensor<[50, 1024]> self = ?,<br>List[int] size = [50, 1, 1024]                 | Fallback |
| 1112 | Tensor<[50, 2304]> self = ?,<br>List[int] size = [50, 1, 2304]                 | Fallback |
| 1113 | Tensor<[50, 3072]> self = ?,<br>List[int] size = [1, 50, 3072]                 | Done     |
| 1114 | Tensor<[50, 3072]> self = ?,<br>List[int] size = [50, 1, 3072]                 | Fallback |
| 1115 | Tensor<[50, 4096]> self = ?,<br>List[int] size = [1, 50, 4096]                 | Done     |
| 1116 | Tensor<[50, 768]> self = ?,<br>List[int] size = [1, 50, 768]                   | Done     |
| 1117 | Tensor<[50, 768]> self = ?,<br>List[int] size = [50, 1, 768]                   | Fallback |
| 1118 | Tensor<[50]> self = ?,<br>List[int] size = [-1, 1]                             | Unknown  |
| 1119 | Tensor<[512]> self = ?,<br>List[int] size = [1, -1, 1, 1]                      | Fallback |
| 1120 | Tensor<[56, 56]> self = ?,<br>List[int] size = [8, 7, 8, 7]                    | Fallback |
| 1121 | Tensor<[59, 1024]> self = ?,<br>List[int] size = [1, 59, 1024]                 | Done     |
| 1122 | Tensor<[59, 50272]> self = ?,<br>List[int] size = [1, 59, 50272]               | Done     |
| 1123 | Tensor<[59, 512]> self = ?,<br>List[int] size = [1, 59, 512]                   | Done     |
| 1124 | Tensor<[5]> self = ?,<br>List[int] size = [-1, 1]                              | Fallback |
| 1125 | Tensor<[5]> self = ?,<br>List[int] size = [1, -1]                              | Fallback |
| 1126 | Tensor<[6, 1, 100, 256]> self = ?,<br>List[int] size = [600, 256]              | Unknown  |
| 1127 | Tensor<[6, 1, 15]> self = ?,<br>List[int] size = [1, 6, 1, 15]                 | Unknown  |
| 1128 | Tensor<[6, 1, 17]> self = ?,<br>List[int] size = [1, 6, 1, 17]                 | Unknown  |
| 1129 | Tensor<[6, 1, 1]> self = ?,<br>List[int] size = [1, 6, 1, 1]                   | Unknown  |
| 1130 | Tensor<[6, 1, 2]> self = ?,<br>List[int] size = [1, 6, 1, 2]                   | Unknown  |
| 1131 | Tensor<[6, 1, 64]> self = ?,<br>List[int] size = [1, 6, 1, 64]                 | Unknown  |
| 1132 | Tensor<[6, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 6, 1, <s0 + 1>]       | Unknown  |
| 1133 | Tensor<[6, 15, 15]> self = ?,<br>List[int] size = [1, 6, 15, 15]               | Done     |
| 1134 | Tensor<[6, 15, 64]> self = ?,<br>List[int] size = [1, 6, 15, 64]               | Done     |
| 1135 | Tensor<[600, 256]> self = ?,<br>List[int] size = [6, 1, 100, 256]              | Unknown  |
| 1136 | Tensor<[600, 4]> self = ?,<br>List[int] size = [6, 1, 100, 4]                  | Unknown  |
| 1137 | Tensor<[600, 92]> self = ?,<br>List[int] size = [6, 1, 100, 92]                | Unknown  |
| 1138 | Tensor<[63, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                       | Unknown  |
| 1139 | Tensor<[63, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                       | Unknown  |
| 1140 | Tensor<[64, 10240]> self = ?,<br>List[int] size = [1, 64, 10240]               | Done     |
| 1141 | Tensor<[64, 1024]> self = ?,<br>List[int] size = [1, 64, 1024]                 | Done     |
| 1142 | Tensor<[64, 1024]> self = ?,<br>List[int] size = [1, 8, 8, 1024]               | Fallback |
| 1143 | Tensor<[64, 1280]> self = ?,<br>List[int] size = [1, 64, 1280]                 | Done     |
| 1144 | Tensor<[64, 2304]> self = ?,<br>List[int] size = [1, 64, 2304]                 | Done     |
| 1145 | Tensor<[64, 3, 49, 49]> self = ?,<br>List[int] size = [1, 64, 3, 49, 49]       | Fallback |
| 1146 | Tensor<[64, 3, 49, 49]> self = ?,<br>List[int] size = [192, 49, 49]            | Fallback |
| 1147 | Tensor<[64, 3, 64, 64]> self = ?,<br>List[int] size = [1, 64, 3, 64, 64]       | Fallback |
| 1148 | Tensor<[64, 3, 64, 64]> self = ?,<br>List[int] size = [192, 64, 64]            | Fallback |
| 1149 | Tensor<[64, 3072]> self = ?,<br>List[int] size = [1, 64, 3072]                 | Done     |
| 1150 | Tensor<[64, 3072]> self = ?,<br>List[int] size = [1, 8, 8, 3072]               | Fallback |
| 1151 | Tensor<[64, 4, 49, 49]> self = ?,<br>List[int] size = [1, 64, 4, 49, 49]       | Fallback |
| 1152 | Tensor<[64, 4, 49, 49]> self = ?,<br>List[int] size = [256, 49, 49]            | Fallback |
| 1153 | Tensor<[64, 4, 64, 64]> self = ?,<br>List[int] size = [1, 64, 4, 64, 64]       | Fallback |
| 1154 | Tensor<[64, 4, 64, 64]> self = ?,<br>List[int] size = [256, 64, 64]            | Fallback |
| 1155 | Tensor<[64, 4096]> self = ?,<br>List[int] size = [1, 8, 8, 4096]               | Fallback |
| 1156 | Tensor<[64, 49, 128]> self = ?,<br>List[int] size = [1, 8, 8, 7, 7, 128]       | Fallback |
| 1157 | Tensor<[64, 49, 128]> self = ?,<br>List[int] size = [3136, 128]                | Fallback |
| 1158 | Tensor<[64, 49, 288]> self = ?,<br>List[int] size = [64, 49, 3, 3, 32]         | Fallback |
| 1159 | Tensor<[64, 49, 32]> self = ?,<br>List[int] size = [4, 16, 49, 32]             | Fallback |
| 1160 | Tensor<[64, 49, 384]> self = ?,<br>List[int] size = [64, 49, 3, 4, 32]         | Fallback |
| 1161 | Tensor<[64, 49, 49]> self = ?,<br>List[int] size = [4, 16, 49, 49]             | Fallback |
| 1162 | Tensor<[64, 49, 96]> self = ?,<br>List[int] size = [1, 8, 8, 7, 7, 96]         | Fallback |
| 1163 | Tensor<[64, 49, 96]> self = ?,<br>List[int] size = [3136, 96]                  | Fallback |
| 1164 | Tensor<[64, 64, 128]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 128]       | Fallback |
| 1165 | Tensor<[64, 64, 128]> self = ?,<br>List[int] size = [4096, 128]                | Fallback |
| 1166 | Tensor<[64, 64, 288]> self = ?,<br>List[int] size = [64, 64, 3, 3, 32]         | Fallback |
| 1167 | Tensor<[64, 64, 32]> self = ?,<br>List[int] size = [4, 16, 64, 32]             | Fallback |
| 1168 | Tensor<[64, 64, 384]> self = ?,<br>List[int] size = [64, 64, 3, 4, 32]         | Fallback |
| 1169 | Tensor<[64, 64, 64]> self = ?,<br>List[int] size = [4, 16, 64, 64]             | Fallback |
| 1170 | Tensor<[64, 64, 96]> self = ?,<br>List[int] size = [1, 8, 8, 8, 8, 96]         | Fallback |
| 1171 | Tensor<[64, 64, 96]> self = ?,<br>List[int] size = [4096, 96]                  | Fallback |
| 1172 | Tensor<[64, 64]> self = ?,<br>List[int] size = [8, 8, 8, 8]                    | Fallback |
| 1173 | Tensor<[64, 768]> self = ?,<br>List[int] size = [1, 64, 768]                   | Done     |
| 1174 | Tensor<[64, 768]> self = ?,<br>List[int] size = [1, 8, 8, 768]                 | Fallback |
| 1175 | Tensor<[64, 9, 64]> self = ?,<br>List[int] size = [1, 64, 9, 64]               | Done     |
| 1176 | Tensor<[64, 9, 9]> self = ?,<br>List[int] size = [1, 64, 9, 9]                 | Done     |
| 1177 | Tensor<[64]> self = ?,<br>List[int] size = [1, -1, 1, 1]                       | Unknown  |
| 1178 | Tensor<[68]> self = ?,<br>List[int] size = [1, -1]                             | Unknown  |
| 1179 | Tensor<[7, 18176]> self = ?,<br>List[int] size = [1, 7, 18176]                 | Unknown  |
| 1180 | Tensor<[7, 2304]> self = ?,<br>List[int] size = [1, 7, 2304]                   | Unknown  |
| 1181 | Tensor<[7, 2]> self = ?,<br>List[int] size = [1, 7, 2]                         | Unknown  |
| 1182 | Tensor<[7, 3072]> self = ?,<br>List[int] size = [1, 7, 3072]                   | Unknown  |
| 1183 | Tensor<[7, 4544]> self = ?,<br>List[int] size = [1, 7, 4544]                   | Unknown  |
| 1184 | Tensor<[7, 4672]> self = ?,<br>List[int] size = [1, 7, 4672]                   | Unknown  |
| 1185 | Tensor<[7, 65024]> self = ?,<br>List[int] size = [1, 7, 65024]                 | Unknown  |
| 1186 | Tensor<[7, 768]> self = ?,<br>List[int] size = [1, 7, 768]                     | Unknown  |
| 1187 | Tensor<[71, 7, 64]> self = ?,<br>List[int] size = [1, 71, 7, 64]               | Unknown  |
| 1188 | Tensor<[71, 7, 7]> self = ?,<br>List[int] size = [1, 71, 7, 7]                 | Unknown  |
| 1189 | Tensor<[768, 196]> self = ?,<br>List[int] size = [1, 768, 196]                 | Fallback |
| 1190 | Tensor<[768, 384]> self = ?,<br>List[int] size = [1, 768, 384]                 | Done     |
| 1191 | Tensor<[784, 1024]> self = ?,<br>List[int] size = [1, 28, 28, 1024]            | Fallback |
| 1192 | Tensor<[784, 192]> self = ?,<br>List[int] size = [1, 28, 28, 192]              | Fallback |
| 1193 | Tensor<[784, 192]> self = ?,<br>List[int] size = [16, 49, 192]                 | Fallback |
| 1194 | Tensor<[784, 256]> self = ?,<br>List[int] size = [1, 28, 28, 256]              | Fallback |
| 1195 | Tensor<[784, 256]> self = ?,<br>List[int] size = [16, 49, 256]                 | Fallback |
| 1196 | Tensor<[784, 576]> self = ?,<br>List[int] size = [16, 49, 576]                 | Fallback |
| 1197 | Tensor<[784, 768]> self = ?,<br>List[int] size = [1, 28, 28, 768]              | Fallback |
| 1198 | Tensor<[784, 768]> self = ?,<br>List[int] size = [16, 49, 768]                 | Fallback |
| 1199 | Tensor<[7]> self = ?,<br>List[int] size = [-1, 1]                              | Unknown  |
| 1200 | Tensor<[8, 1, 10]> self = ?,<br>List[int] size = [1, 8, 1, 10]                 | Unknown  |
| 1201 | Tensor<[8, 1, 1]> self = ?,<br>List[int] size = [1, 8, 1, 1]                   | Unknown  |
| 1202 | Tensor<[8, 1, 2]> self = ?,<br>List[int] size = [1, 8, 1, 2]                   | Unknown  |
| 1203 | Tensor<[8, 1, 64]> self = ?,<br>List[int] size = [1, 8, 1, 64]                 | Unknown  |
| 1204 | Tensor<[8, 1, s0 + 1]> self = ?,<br>List[int] size = [1, 8, 1, <s0 + 1>]       | Unknown  |
| 1205 | Tensor<[8, 10, 10]> self = ?,<br>List[int] size = [1, 8, 10, 10]               | Done     |
| 1206 | Tensor<[8, 10, 64]> self = ?,<br>List[int] size = [1, 8, 10, 64]               | Done     |
| 1207 | Tensor<[8, 2048, 256]> self = ?,<br>List[int] size = [1, 8, 2048, 256]         | Unknown  |
| 1208 | Tensor<[8, 2048, 96]> self = ?,<br>List[int] size = [1, 8, 2048, 96]           | Unknown  |
| 1209 | Tensor<[8, 256, 160]> self = ?,<br>List[int] size = [1, 8, 256, 160]           | Unknown  |
| 1210 | Tensor<[8, 256, 2048]> self = ?,<br>List[int] size = [1, 8, 256, 2048]         | Unknown  |
| 1211 | Tensor<[8, 256, 256]> self = ?,<br>List[int] size = [1, 8, 256, 256]           | Done     |
| 1212 | Tensor<[8, 256, 32]> self = ?,<br>List[int] size = [1, 8, 256, 32]             | Done     |
| 1213 | Tensor<[8, 300, 300]> self = ?,<br>List[int] size = [1, 8, 300, 300]           | Fallback |
| 1214 | Tensor<[8, 300, 64]> self = ?,<br>List[int] size = [1, 8, 300, 64]             | Done     |
| 1215 | Tensor<[850, 4]> self = ?,<br>List[int] size = [-1, 1, 4]                      | Unknown  |
| 1216 | Tensor<[850, 9, 4]> self = ?,<br>List[int] size = [-1, 4]                      | Unknown  |
| 1217 | Tensor<[8732, 1, 4]> self = ?,<br>List[int] size = [8732, 4]                   | Unknown  |
| 1218 | Tensor<[8732, 2, 2]> self = ?,<br>List[int] size = [8732, 4]                   | Unknown  |
| 1219 | Tensor<[9, 1024]> self = ?,<br>List[int] size = [1, 9, 1024]                   | Done     |
| 1220 | Tensor<[9, 1280]> self = ?,<br>List[int] size = [1, 9, 1280]                   | Done     |
| 1221 | Tensor<[9, 128]> self = ?,<br>List[int] size = [1, 9, 128]                     | Done     |
| 1222 | Tensor<[9, 12]> self = ?,<br>List[int] size = [-1, 2]                          | Fallback |
| 1223 | Tensor<[9, 16384]> self = ?,<br>List[int] size = [1, 9, 16384]                 | Done     |
| 1224 | Tensor<[9, 2048]> self = ?,<br>List[int] size = [1, 9, 2048]                   | Done     |
| 1225 | Tensor<[9, 30000]> self = ?,<br>List[int] size = [1, 9, 30000]                 | Fallback |
| 1226 | Tensor<[9, 3072]> self = ?,<br>List[int] size = [1, 9, 3072]                   | Done     |
| 1227 | Tensor<[9, 320]> self = ?,<br>List[int] size = [1, 9, 320]                     | Done     |
| 1228 | Tensor<[9, 4096]> self = ?,<br>List[int] size = [1, 9, 4096]                   | Done     |
| 1229 | Tensor<[9, 4]> self = ?,<br>List[int] size = [1, -1, 4]                        | Unknown  |
| 1230 | Tensor<[9, 640]> self = ?,<br>List[int] size = [1, 9, 640]                     | Done     |
| 1231 | Tensor<[9, 768]> self = ?,<br>List[int] size = [1, 9, 768]                     | Done     |
| 1232 | Tensor<[9, 8192]> self = ?,<br>List[int] size = [1, 9, 8192]                   | Done     |
| 1233 | Tensor<[9, 8]> self = ?,<br>List[int] size = [-1, 2]                           | Fallback |
| 1234 | Tensor<[920, 1, 2048]> self = ?,<br>List[int] size = [920, 2048]               | Unknown  |
| 1235 | Tensor<[920, 1, 256]> self = ?,<br>List[int] size = [920, 256]                 | Unknown  |
| 1236 | Tensor<[920, 1, 256]> self = ?,<br>List[int] size = [920, 8, 32]               | Unknown  |
| 1237 | Tensor<[920, 2048]> self = ?,<br>List[int] size = [920, 1, 2048]               | Unknown  |
| 1238 | Tensor<[920, 256]> self = ?,<br>List[int] size = [920, 1, 256]                 | Unknown  |
| 1239 | Tensor<[920, 8, 32]> self = ?,<br>List[int] size = [920, 256]                  | Unknown  |
| 1240 | Tensor<[96, 49, 32]> self = ?,<br>List[int] size = [16, 6, 49, 32]             | Fallback |
| 1241 | Tensor<[96, 49, 49]> self = ?,<br>List[int] size = [16, 6, 49, 49]             | Fallback |
| 1242 | Tensor<[96, 64, 32]> self = ?,<br>List[int] size = [16, 6, 64, 32]             | Fallback |
| 1243 | Tensor<[96, 64, 64]> self = ?,<br>List[int] size = [16, 6, 64, 64]             | Fallback |
| 1244 | Tensor<[9]> self = ?,<br>List[int] size = [1, -1]                              | Unknown  |
| 1245 | Tensor<[s0, 256]> self = ?,<br>List[int] size = [1, <s0>, 256]                 | Unknown  |
| 1246 | Tensor<[s0, 768]> self = ?,<br>List[int] size = [1, <s0>, 768]                 | Unknown  |
