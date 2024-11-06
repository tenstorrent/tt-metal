# look for | Unknown  | Unknown    | N/A   | and isolate those s0 shapes
data = """
|   8 | Tensor<[1, 1, 1024]> self = , Tensor<[1, 1, 1024]> other = ?              | Unknown  | Done       | True  |
|   9 | Tensor<[1, 1, 16, 32]> self = , Tensor<[1, 1, 16, 32]> other = ?          | Unknown  | Done       | True  |
|  54 | Tensor<[1, 1, 3072]> self = , Tensor<[1, 1, 3072]> other = ?              | Unknown  | Done       | True  |
|  61 | Tensor<[1, 1, 4096]> self = , Tensor<[1, 1, 4096]> other = ?              | Unknown  | Done       | True  |
|  63 | Tensor<[1, 1, 512]> self = , Tensor<[1, 1, 512]> other = ?                | Unknown  | Done       | True  |
|  64 | Tensor<[1, 1, 7, 64]> self = , Tensor<[1, 1, 7, 64]> other = ?            | Done     | Done       | True  |
|  65 | Tensor<[1, 1, 768]> self = , Tensor<[1, 1, 768]> other = ?                | Done     | Done       | True  |
|  66 | Tensor<[1, 1, 768]> self = , Tensor<[1, 768]> other = ?                   | Done     | Done       | True  |
|  67 | Tensor<[1, 10, 1024]> self = , Tensor<[1, 10, 1024]> other = ?            | Done     | Done       | True  |
|  69 | Tensor<[1, 10, 512]> self = , Tensor<[1, 10, 512]> other = ?              | Done     | Done       | True  |
|  70 | Tensor<[1, 10, 768]> self = , Tensor<[1, 10, 768]> other = ?              | Done     | Done       | True  |
|  71 | Tensor<[1, 100, 14, 14]> self = , Tensor<[1, 100, 14, 14]> other = ?      | Done     | Done       | True  |
|  72 | Tensor<[1, 1008, 7, 7]> self = , Tensor<[1, 1008, 7, 7]> other = ?        | Done     | Done       | True  |
|  75 | Tensor<[1, 1024, 10, 10]> self = , Tensor<[1, 1024, 10, 10]> other = ?    | Done     | Done       | True  |
|  76 | Tensor<[1, 1024, 14, 14]> self = , Tensor<[1, 1024, 14, 14]> other = ?    | Done     | Done       | True  |
|  77 | Tensor<[1, 1024, 16, 16]> self = , Tensor<[1, 1024, 16, 16]> other = ?    | Done     | Done       | True  |
|  78 | Tensor<[1, 1024, 160]> self = , Tensor<[1, 1024, 160]> other = ?          | Done     | Done       | True  |
|  79 | Tensor<[1, 1024, 17, 17]> self = , Tensor<[1, 1024, 17, 17]> other = ?    | Done     | Done       | True  |
|  80 | Tensor<[1, 1024, 256]> self = , Tensor<[256]> other = ?                   | Done     | Done       | True  |
|  81 | Tensor<[1, 1024, 45, 80]> self = , Tensor<[1, 1024, 1, 1]> other = ?      | Done     | Done       | True  |
|  82 | Tensor<[1, 1024, 45, 80]> self = , Tensor<[1, 1024, 45, 80]> other = ?    | Done     | Done       | True  |
|  83 | Tensor<[1, 1024, 50, 68]> self = , Tensor<[1, 1024, 1, 1]> other = ?      | Unknown  | Done       | True  |
|  84 | Tensor<[1, 1024, 50, 68]> self = , Tensor<[1, 1024, 50, 68]> other = ?    | Unknown  | Done       | True  |
|  85 | Tensor<[1, 1024, 7, 7]> self = , Tensor<[1, 1024, 7, 7]> other = ?        | Done     | Done       | True  |
|  86 | Tensor<[1, 1024]> self = , Tensor<[1, 1024]> other = ?                    | Unknown  | Done       | True  |
|  87 | Tensor<[1, 104, 28, 28]> self = , Tensor<[1, 104, 28, 28]> other = ?      | Done     | Done       | True  |
|  88 | Tensor<[1, 1056, 48, 48]> self = , Tensor<[1, 1056, 48, 48]> other = ?    | Done     | Done       | True  |
|  91 | Tensor<[1, 112, 14, 14]> self = , Tensor<[1, 112, 14, 14]> other = ?      | Done     | Done       | True  |
|  92 | Tensor<[1, 112, 15, 15]> self = , Tensor<[1, 112, 15, 15]> other = ?      | Done     | Done       | True  |
|  93 | Tensor<[1, 112, 20, 20]> self = , Tensor<[1, 112, 20, 20]> other = ?      | Done     | Done       | True  |
|  94 | Tensor<[1, 112, 24, 24]> self = , Tensor<[1, 112, 24, 24]> other = ?      | Done     | Done       | True  |
|  95 | Tensor<[1, 116, 14, 14]> self = , Tensor<[1, 116, 14, 14]> other = ?      | Done     | Done       | True  |
|  96 | Tensor<[1, 12, 1, 10]> self = , Tensor<[1, 1, 1, 10]> other = ?           | Unknown  | Fallback   | True  |
|  97 | Tensor<[1, 12, 1, 10]> self = , Tensor<[1, 12, 1, 10]> other = ?          | Unknown  | Done       | True  |
|  98 | Tensor<[1, 12, 1, 1]> self = , Tensor<[1, 1, 1, 1]> other = ?             | Unknown  | Fallback   | True  |
|  99 | Tensor<[1, 12, 1, 1]> self = , Tensor<[1, 12, 1, 1]> other = ?            | Unknown  | Done       | True  |
| 100 | Tensor<[1, 12, 1, 2]> self = , Tensor<[1, 1, 1, 2]> other = ?             | Unknown  | Fallback   | True  |
| 101 | Tensor<[1, 12, 1, 2]> self = , Tensor<[1, 12, 1, 2]> other = ?            | Unknown  | Done       | True  |
| 102 | Tensor<[1, 12, 1, 46]> self = , Tensor<[1, 1, 1, 46]> other = ?           | Unknown  | Fallback   | True  |
| 106 | Tensor<[1, 12, 10, 10]> self = , Tensor<[1, 1, 1, 10]> other = ?          | Done     | Done       | True  |
| 107 | Tensor<[1, 12, 10, 10]> self = , Tensor<[1, 12, 10, 10]> other = ?        | Done     | Done       | True  |
| 108 | Tensor<[1, 12, 12, 12]> self = , Tensor<[1, 1, 1, 12]> other = ?          | Done     | Done       | True  |
| 109 | Tensor<[1, 12, 128]> self = , Tensor<[1, 12, 128]> other = ?              | Done     | Done       | True  |
| 110 | Tensor<[1, 12, 14, 14]> self = , Tensor<[1, 1, 1, 14]> other = ?          | Done     | Done       | True  |
| 111 | Tensor<[1, 12, 197, 197]> self = , Tensor<[1, 12, 197, 197]> other = ?    | Done     | Done       | True  |
| 112 | Tensor<[1, 12, 24, 24]> self = , Tensor<[1, 1, 24, 24]> other = ?         | None     | Fallback   | True  |
| 113 | Tensor<[1, 12, 25, 25]> self = , Tensor<[1, 1, 1, 25]> other = ?          | Done     | Done       | True  |
| 115 | Tensor<[1, 12, 3072]> self = , Tensor<[1, 12, 3072]> other = ?            | Done     | Done       | True  |
| 116 | Tensor<[1, 12, 45, 45]> self = , Tensor<[1, 1, 45, 45]> other = ?         | Unknown  | Fallback   | True  |
| 117 | Tensor<[1, 12, 56, 56]> self = , Tensor<[1, 12, 56, 56]> other = ?        | Done     | Done       | True  |
| 118 | Tensor<[1, 12, 7, 7]> self = , Tensor<[1, 1, 1, 7]> other = ?             | Done     | Done       | True  |
| 119 | Tensor<[1, 12, 768]> self = , Tensor<[1, 12, 768]> other = ?              | None     | Fallback   | True  |
| 120 | Tensor<[1, 12, 9, 9]> self = , Tensor<[1, 1, 1, 9]> other = ?             | Done     | Done       | True  |
| 121 | Tensor<[1, 120, 17, 17]> self = , Tensor<[1, 120, 17, 17]> other = ?      | Done     | Done       | True  |
| 122 | Tensor<[1, 120, 28, 28]> self = , Tensor<[1, 120, 28, 28]> other = ?      | Done     | Done       | True  |
| 123 | Tensor<[1, 1200, 320]> self = , Tensor<[1, 1200, 320]> other = ?          | Done     | Done       | True  |
| 124 | Tensor<[1, 1232, 14, 14]> self = , Tensor<[1, 1232, 14, 14]> other = ?    | Done     | Done       | True  |
| 127 | Tensor<[1, 128, 100, 136]> self = , Tensor<[1, 128, 1, 1]> other = ?      | Unknown  | Done       | True  |
| 128 | Tensor<[1, 128, 112, 112]> self = , Tensor<[1, 128, 112, 112]> other = ?  | Done     | Done       | True  |
| 129 | Tensor<[1, 128, 128, 128]> self = , Tensor<[1, 128, 128, 128]> other = ?  | Done     | Done       | True  |
| 130 | Tensor<[1, 128, 180, 320]> self = , Tensor<[1, 128, 1, 1]> other = ?      | Done     | Done       | True  |
| 131 | Tensor<[1, 128, 200, 272]> self = , Tensor<[1, 128, 1, 1]> other = ?      | Unknown  | Done       | True  |
| 132 | Tensor<[1, 128, 28, 28]> self = , Tensor<[1, 128, 28, 28]> other = ?      | Done     | Done       | True  |
| 133 | Tensor<[1, 128, 56, 56]> self = , Tensor<[1, 128, 56, 56]> other = ?      | Done     | Done       | True  |
| 134 | Tensor<[1, 128, 64, 64]> self = , Tensor<[1, 128, 64, 64]> other = ?      | Done     | Done       | True  |
| 135 | Tensor<[1, 128, 75, 75]> self = , Tensor<[1, 128, 75, 75]> other = ?      | Done     | Done       | True  |
| 136 | Tensor<[1, 128, 90, 160]> self = , Tensor<[1, 128, 1, 1]> other = ?       | Done     | Done       | True  |
| 137 | Tensor<[1, 1280, 8, 8]> self = , Tensor<[1, 1280, 1, 1]> other = ?        | Unknown  | Done       | True  |
| 138 | Tensor<[1, 1280, 8, 8]> self = , Tensor<[1, 1280, 8, 8]> other = ?        | Unknown  | Done       | True  |
| 143 | Tensor<[1, 1344, 14, 14]> self = , Tensor<[1, 1344, 14, 14]> other = ?    | Done     | Done       | True  |
| 144 | Tensor<[1, 136, 19, 19]> self = , Tensor<[1, 136, 19, 19]> other = ?      | Done     | Done       | True  |
| 145 | Tensor<[1, 1370, 1280]> self = , Tensor<[1, 1370, 1280]> other = ?        | Done     | Done       | True  |
| 146 | Tensor<[1, 1392, 14, 14]> self = , Tensor<[1, 1392, 14, 14]> other = ?    | Done     | Done       | True  |
| 147 | Tensor<[1, 14, 128]> self = , Tensor<[1, 14, 128]> other = ?              | Done     | Done       | True  |
| 148 | Tensor<[1, 14, 14, 384]> self = , Tensor<[1, 14, 14, 384]> other = ?      | Done     | Done       | True  |
| 149 | Tensor<[1, 14, 14, 512]> self = , Tensor<[1, 14, 14, 512]> other = ?      | Done     | Done       | True  |
| 151 | Tensor<[1, 14, 3072]> self = , Tensor<[1, 14, 3072]> other = ?            | Done     | Done       | True  |
| 152 | Tensor<[1, 14, 56, 56]> self = , Tensor<[1, 14, 56, 56]> other = ?        | Done     | Done       | True  |
| 153 | Tensor<[1, 14, 768]> self = , Tensor<[1, 14, 768]> other = ?              | None     | Fallback   | True  |
| 154 | Tensor<[1, 144, 28, 28]> self = , Tensor<[1, 144, 28, 28]> other = ?      | Done     | Done       | True  |
| 155 | Tensor<[1, 144, 7, 7]> self = , Tensor<[1, 144, 7, 7]> other = ?          | Done     | Done       | True  |
| 156 | Tensor<[1, 1445, 192]> self = , Tensor<[1, 1445, 192]> other = ?          | Done     | Done       | True  |
| 158 | Tensor<[1, 15, 1024]> self = , Tensor<[1, 15, 1024]> other = ?            | Done     | Done       | True  |
| 160 | Tensor<[1, 15, 512]> self = , Tensor<[1, 15, 512]> other = ?              | Done     | Done       | True  |
| 161 | Tensor<[1, 1500, 768]> self = , Tensor<[1, 1500, 768]> other = ?          | Done     | Done       | True  |
| 162 | Tensor<[1, 1500, 768]> self = , Tensor<[1500, 768]> other = ?             | Done     | Done       | True  |
| 163 | Tensor<[1, 1512, 7, 7]> self = , Tensor<[1, 1512, 7, 7]> other = ?        | Done     | Done       | True  |
| 164 | Tensor<[1, 1536, 8, 8]> self = , Tensor<[1, 1536, 8, 8]> other = ?        | Done     | Done       | True  |
| 165 | Tensor<[1, 16, 1, 10]> self = , Tensor<[1, 1, 1, 10]> other = ?           | Unknown  | Fallback   | True  |
| 166 | Tensor<[1, 16, 1, 10]> self = , Tensor<[1, 16, 1, 10]> other = ?          | Unknown  | Done       | True  |
| 167 | Tensor<[1, 16, 1, 1]> self = , Tensor<[1, 1, 1, 1]> other = ?             | Unknown  | Fallback   | True  |
| 168 | Tensor<[1, 16, 1, 1]> self = , Tensor<[1, 16, 1, 1]> other = ?            | Unknown  | Done       | True  |
| 169 | Tensor<[1, 16, 1, 2]> self = , Tensor<[1, 1, 1, 2]> other = ?             | Unknown  | Fallback   | True  |
| 170 | Tensor<[1, 16, 1, 2]> self = , Tensor<[1, 16, 1, 2]> other = ?            | Unknown  | Done       | True  |
| 171 | Tensor<[1, 16, 1, 60]> self = , Tensor<[1, 1, 1, 60]> other = ?           | Unknown  | Fallback   | True  |
| 172 | Tensor<[1, 16, 1, 6]> self = , Tensor<[1, 1, 1, 6]> other = ?             | Unknown  | Fallback   | True  |
| 176 | Tensor<[1, 16, 10, 10]> self = , Tensor<[1, 1, 1, 10]> other = ?          | Done     | Done       | True  |
| 177 | Tensor<[1, 16, 10, 10]> self = , Tensor<[1, 16, 10, 10]> other = ?        | Done     | Done       | True  |
| 178 | Tensor<[1, 16, 112, 112]> self = , Tensor<[1, 16, 112, 112]> other = ?    | Done     | Done       | True  |
| 179 | Tensor<[1, 16, 16, 384]> self = , Tensor<[1, 16, 16, 384]> other = ?      | Done     | Done       | True  |
| 180 | Tensor<[1, 16, 16, 512]> self = , Tensor<[1, 16, 16, 512]> other = ?      | Done     | Done       | True  |
| 181 | Tensor<[1, 16, 160, 160]> self = , Tensor<[1, 16, 160, 160]> other = ?    | Done     | Done       | True  |
| 182 | Tensor<[1, 16, 19, 19]> self = , Tensor<[1, 1, 19, 19]> other = ?         | None     | Fallback   | True  |
| 183 | Tensor<[1, 16, 197, 197]> self = , Tensor<[1, 16, 197, 197]> other = ?    | Done     | Done       | True  |
| 184 | Tensor<[1, 16, 256, 256]> self = , Tensor<[1, 1, 1, 256]> other = ?       | Done     | Done       | True  |
| 185 | Tensor<[1, 16, 28, 28]> self = , Tensor<[1, 16, 28, 28]> other = ?        | Done     | Done       | True  |
| 186 | Tensor<[1, 16, 5, 5]> self = , Tensor<[1, 1, 1, 5]> other = ?             | Unknown  | Done       | True  |
| 187 | Tensor<[1, 16, 59, 59]> self = , Tensor<[1, 1, 59, 59]> other = ?         | None     | Fallback   | True  |
| 188 | Tensor<[1, 16, 6, 49, 49]> self = , Tensor<[1, 16, 1, 49, 49]> other = ?  | None     | Fallback   | True  |
| 189 | Tensor<[1, 16, 6, 64, 64]> self = , Tensor<[1, 16, 1, 64, 64]> other = ?  | None     | Fallback   | True  |
| 190 | Tensor<[1, 16, 768]> self = , Tensor<[1, 16, 768]> other = ?              | Done     | Done       | True  |
| 191 | Tensor<[1, 16, 8, 49, 49]> self = , Tensor<[1, 16, 1, 49, 49]> other = ?  | None     | Fallback   | True  |
| 192 | Tensor<[1, 16, 8, 64, 64]> self = , Tensor<[1, 16, 1, 64, 64]> other = ?  | None     | Fallback   | True  |
| 193 | Tensor<[1, 16, 9, 9]> self = , Tensor<[1, 1, 1, 9]> other = ?             | Done     | Done       | True  |
| 194 | Tensor<[1, 160, 14, 14]> self = , Tensor<[1, 160, 14, 14]> other = ?      | Done     | Done       | True  |
| 195 | Tensor<[1, 160, 24, 24]> self = , Tensor<[1, 160, 24, 24]> other = ?      | Done     | Done       | True  |
| 196 | Tensor<[1, 160, 28, 28]> self = , Tensor<[1, 160, 28, 28]> other = ?      | Done     | Done       | True  |
| 197 | Tensor<[1, 160, 32, 32]> self = , Tensor<[1, 160, 32, 32]> other = ?      | Unknown  | Done       | True  |
| 198 | Tensor<[1, 160, 7, 7]> self = , Tensor<[1, 160, 7, 7]> other = ?          | Done     | Done       | True  |
| 199 | Tensor<[1, 160, 73, 73]> self = , Tensor<[1, 160, 73, 73]> other = ?      | Done     | Done       | True  |
| 200 | Tensor<[1, 16384, 256]> self = , Tensor<[256]> other = ?                  | Done     | Done       | True  |
| 201 | Tensor<[1, 16384, 32]> self = , Tensor<[1, 16384, 32]> other = ?          | Done     | Done       | True  |
| 202 | Tensor<[1, 168, 28, 28]> self = , Tensor<[1, 168, 28, 28]> other = ?      | Done     | Done       | True  |
| 203 | Tensor<[1, 18, 56, 56]> self = , Tensor<[1, 18, 56, 56]> other = ?        | Done     | Done       | True  |
| 204 | Tensor<[1, 185, 28, 28]> self = , Tensor<[1, 185, 28, 28]> other = ?      | Done     | Done       | True  |
| 205 | Tensor<[1, 19, 1024]> self = , Tensor<[1, 19, 1024]> other = ?            | Done     | Done       | True  |
| 206 | Tensor<[1, 192, 14, 14]> self = , Tensor<[1, 192, 14, 14]> other = ?      | Done     | Done       | True  |
| 207 | Tensor<[1, 192, 28, 28]> self = , Tensor<[1, 192, 28, 28]> other = ?      | Done     | Done       | True  |
| 208 | Tensor<[1, 192, 32, 42]> self = , Tensor<[1, 192, 32, 42]> other = ?      | Done     | Done       | True  |
| 209 | Tensor<[1, 192, 7, 7]> self = , Tensor<[1, 192, 7, 7]> other = ?          | Done     | Done       | True  |
| 210 | Tensor<[1, 192, 71, 71]> self = , Tensor<[1, 192, 71, 71]> other = ?      | Done     | Done       | True  |
| 211 | Tensor<[1, 192, 8, 8]> self = , Tensor<[1, 192, 8, 8]> other = ?          | Done     | Done       | True  |
| 212 | Tensor<[1, 1920, 7, 7]> self = , Tensor<[1, 1920, 7, 7]> other = ?        | Done     | Done       | True  |
| 213 | Tensor<[1, 19200, 64]> self = , Tensor<[1, 19200, 64]> other = ?          | Done     | Done       | True  |
| 214 | Tensor<[1, 196, 14, 14]> self = , Tensor<[1, 196, 14, 14]> other = ?      | Done     | Done       | True  |
| 215 | Tensor<[1, 196, 768]> self = , Tensor<[1, 196, 768]> other = ?            | None     | Fallback   | True  |
| 216 | Tensor<[1, 197, 1024]> self = , Tensor<[1, 197, 1024]> other = ?          | Done     | Done       | True  |
| 217 | Tensor<[1, 197, 768]> self = , Tensor<[1, 197, 768]> other = ?            | Done     | Done       | True  |
| 222 | Tensor<[1, 20, 28, 28]> self = , Tensor<[1, 20, 28, 28]> other = ?        | Done     | Done       | True  |
| 223 | Tensor<[1, 2016, 7, 7]> self = , Tensor<[1, 2016, 7, 7]> other = ?        | Done     | Done       | True  |
| 226 | Tensor<[1, 2048, 23, 40]> self = , Tensor<[1, 2048, 1, 1]> other = ?      | Done     | Done       | True  |
| 227 | Tensor<[1, 2048, 23, 40]> self = , Tensor<[1, 2048, 23, 40]> other = ?    | Done     | Done       | True  |
| 228 | Tensor<[1, 2048, 25, 34]> self = , Tensor<[1, 2048, 1, 1]> other = ?      | Unknown  | Done       | True  |
| 229 | Tensor<[1, 2048, 25, 34]> self = , Tensor<[1, 2048, 25, 34]> other = ?    | Unknown  | Done       | True  |
| 230 | Tensor<[1, 2048, 7, 7]> self = , Tensor<[1, 2048, 7, 7]> other = ?        | Done     | Done       | True  |
| 231 | Tensor<[1, 2048, 768]> self = , Tensor<[1, 2048, 768]> other = ?          | Done     | Done       | True  |
| 232 | Tensor<[1, 2048, 768]> self = , Tensor<[2048, 768]> other = ?             | Done     | Done       | True  |
| 233 | Tensor<[1, 208, 14, 14]> self = , Tensor<[1, 208, 14, 14]> other = ?      | Done     | Done       | True  |
| 234 | Tensor<[1, 208, 9, 9]> self = , Tensor<[1, 208, 9, 9]> other = ?          | Done     | Done       | True  |
| 235 | Tensor<[1, 216, 28, 28]> self = , Tensor<[1, 216, 28, 28]> other = ?      | Done     | Done       | True  |
| 236 | Tensor<[1, 224, 56, 56]> self = , Tensor<[1, 224, 56, 56]> other = ?      | Done     | Done       | True  |
| 237 | Tensor<[1, 224, 7, 7]> self = , Tensor<[1, 224, 7, 7]> other = ?          | Done     | Done       | True  |
| 239 | Tensor<[1, 232, 10, 10]> self = , Tensor<[1, 232, 10, 10]> other = ?      | Done     | Done       | True  |
| 240 | Tensor<[1, 232, 56, 56]> self = , Tensor<[1, 232, 56, 56]> other = ?      | Done     | Done       | True  |
| 241 | Tensor<[1, 24, 112, 112]> self = , Tensor<[1, 24, 112, 112]> other = ?    | Done     | Done       | True  |
| 242 | Tensor<[1, 24, 28, 28]> self = , Tensor<[1, 24, 28, 28]> other = ?        | Done     | Done       | True  |
| 243 | Tensor<[1, 24, 49, 49]> self = , Tensor<[1, 24, 49, 49]> other = ?        | Done     | Done       | True  |
| 244 | Tensor<[1, 24, 56, 56]> self = , Tensor<[1, 24, 56, 56]> other = ?        | Done     | Done       | True  |
| 245 | Tensor<[1, 24, 60, 60]> self = , Tensor<[1, 24, 60, 60]> other = ?        | Done     | Done       | True  |
| 246 | Tensor<[1, 24, 64, 64]> self = , Tensor<[1, 24, 64, 64]> other = ?        | Done     | Done       | True  |
| 247 | Tensor<[1, 24, 65, 65]> self = , Tensor<[1, 24, 65, 65]> other = ?        | Done     | Done       | True  |
| 248 | Tensor<[1, 24, 768]> self = , Tensor<[1, 24, 768]> other = ?              | None     | Fallback   | True  |
| 249 | Tensor<[1, 24, 80, 80]> self = , Tensor<[1, 24, 80, 80]> other = ?        | Done     | Done       | True  |
| 250 | Tensor<[1, 240, 14, 14]> self = , Tensor<[1, 240, 14, 14]> other = ?      | Done     | Done       | True  |
| 251 | Tensor<[1, 240, 28, 28]> self = , Tensor<[1, 240, 28, 28]> other = ?      | Done     | Done       | True  |
| 252 | Tensor<[1, 25, 768]> self = , Tensor<[1, 25, 768]> other = ?              | Done     | Done       | True  |
| 253 | Tensor<[1, 2520, 7, 7]> self = , Tensor<[1, 2520, 7, 7]> other = ?        | Done     | Done       | True  |
| 256 | Tensor<[1, 256, 100, 136]> self = , Tensor<[1, 256, 1, 1]> other = ?      | Unknown  | Done       | True  |
| 257 | Tensor<[1, 256, 100, 136]> self = , Tensor<[1, 256, 100, 136]> other = ?  | Unknown  | Done       | True  |
| 258 | Tensor<[1, 256, 1024]> self = , Tensor<[1, 256, 1024]> other = ?          | Done     | Done       | True  |
| 259 | Tensor<[1, 256, 128, 128]> self = , Tensor<[1, 256, 128, 128]> other = ?  | Done     | Done       | True  |
| 260 | Tensor<[1, 256, 1280]> self = , Tensor<[1, 256, 1280]> other = ?          | Done     | Done       | True  |
| 261 | Tensor<[1, 256, 14, 14]> self = , Tensor<[1, 256, 14, 14]> other = ?      | Done     | Done       | True  |
| 262 | Tensor<[1, 256, 16, 16]> self = , Tensor<[1, 256, 16, 16]> other = ?      | Unknown  | Done       | True  |
| 263 | Tensor<[1, 256, 160]> self = , Tensor<[1, 256, 160]> other = ?            | Unknown  | Done       | True  |
| 264 | Tensor<[1, 256, 180, 320]> self = , Tensor<[1, 256, 1, 1]> other = ?      | Done     | Done       | True  |
| 265 | Tensor<[1, 256, 180, 320]> self = , Tensor<[1, 256, 180, 320]> other = ?  | Done     | Done       | True  |
| 266 | Tensor<[1, 256, 200, 272]> self = , Tensor<[1, 256, 1, 1]> other = ?      | Unknown  | Done       | True  |
| 267 | Tensor<[1, 256, 200, 272]> self = , Tensor<[1, 256, 200, 272]> other = ?  | Unknown  | Done       | True  |
| 268 | Tensor<[1, 256, 256]> self = , Tensor<[1, 256, 256]> other = ?            | Done     | Done       | True  |
| 269 | Tensor<[1, 256, 256]> self = , Tensor<[256]> other = ?                    | Done     | Done       | True  |
| 270 | Tensor<[1, 256, 28, 28]> self = , Tensor<[1, 256, 28, 28]> other = ?      | Done     | Done       | True  |
| 271 | Tensor<[1, 256, 32, 32]> self = , Tensor<[1, 256, 32, 32]> other = ?      | Done     | Done       | True  |
| 272 | Tensor<[1, 256, 32]> self = , Tensor<[1, 256, 32]> other = ?              | Unknown  | Done       | True  |
| 273 | Tensor<[1, 256, 38, 38]> self = , Tensor<[1, 256, 38, 38]> other = ?      | Done     | Done       | True  |
| 274 | Tensor<[1, 256, 45, 80]> self = , Tensor<[1, 256, 1, 1]> other = ?        | Done     | Done       | True  |
| 275 | Tensor<[1, 256, 50, 68]> self = , Tensor<[1, 256, 1, 1]> other = ?        | Unknown  | Done       | True  |
| 276 | Tensor<[1, 256, 50, 68]> self = , Tensor<[1, 256, 50, 68]> other = ?      | Unknown  | Done       | True  |
| 277 | Tensor<[1, 256, 512]> self = , Tensor<[1, 256, 512]> other = ?            | Done     | Done       | True  |
| 278 | Tensor<[1, 256, 56, 56]> self = , Tensor<[1, 256, 56, 56]> other = ?      | Done     | Done       | True  |
| 279 | Tensor<[1, 256, 64, 64]> self = , Tensor<[1, 256, 64, 64]> other = ?      | Done     | Done       | True  |
| 280 | Tensor<[1, 256, 64]> self = , Tensor<[1, 256, 64]> other = ?              | Unknown  | Done       | True  |
| 281 | Tensor<[1, 256, 7, 7]> self = , Tensor<[1, 256, 7, 7]> other = ?          | Done     | Done       | True  |
| 282 | Tensor<[1, 256, 75, 75]> self = , Tensor<[1, 256, 75, 75]> other = ?      | Done     | Done       | True  |
| 283 | Tensor<[1, 256, 90, 160]> self = , Tensor<[1, 256, 1, 1]> other = ?       | Done     | Done       | True  |
| 284 | Tensor<[1, 272, 12, 12]> self = , Tensor<[1, 272, 12, 12]> other = ?      | Done     | Done       | True  |
| 285 | Tensor<[1, 272, 7, 7]> self = , Tensor<[1, 272, 7, 7]> other = ?          | Done     | Done       | True  |
| 286 | Tensor<[1, 28, 28, 192]> self = , Tensor<[1, 28, 28, 192]> other = ?      | Done     | Done       | True  |
| 287 | Tensor<[1, 28, 28, 256]> self = , Tensor<[1, 28, 28, 256]> other = ?      | Done     | Done       | True  |
| 288 | Tensor<[1, 28, 28, 28]> self = , Tensor<[1, 28, 28, 28]> other = ?        | Done     | Done       | True  |
| 289 | Tensor<[1, 288, 14, 14]> self = , Tensor<[1, 288, 14, 14]> other = ?      | Done     | Done       | True  |
| 290 | Tensor<[1, 2904, 24, 24]> self = , Tensor<[1, 2904, 24, 24]> other = ?    | Done     | Done       | True  |
| 291 | Tensor<[1, 3, 300, 300]> self = , Tensor<[1, 3, 300, 300]> other = ?      | Done     | Done       | True  |
| 292 | Tensor<[1, 3, 320, 320]> self = , Tensor<[1, 3, 320, 320]> other = ?      | Done     | Done       | True  |
| 293 | Tensor<[1, 3, 800, 1066]> self = , Tensor<[1, 3, 800, 1066]> other = ?    | Unknown  | Done       | True  |
| 294 | Tensor<[1, 300, 512]> self = , Tensor<[1, 300, 512]> other = ?            | Done     | Done       | True  |
| 295 | Tensor<[1, 3024, 7, 7]> self = , Tensor<[1, 3024, 7, 7]> other = ?        | Done     | Done       | True  |
| 296 | Tensor<[1, 32, 112, 112]> self = , Tensor<[1, 32, 112, 112]> other = ?    | Done     | Done       | True  |
| 297 | Tensor<[1, 32, 128, 128]> self = , Tensor<[1, 32, 128, 128]> other = ?    | Unknown  | Done       | True  |
| 298 | Tensor<[1, 32, 1536]> self = , Tensor<[1, 32, 1536]> other = ?            | Done     | Done       | True  |
| 299 | Tensor<[1, 32, 256, 256]> self = , Tensor<[1, 32, 256, 256]> other = ?    | Done     | Done       | True  |
| 300 | Tensor<[1, 32, 28, 28]> self = , Tensor<[1, 32, 28, 28]> other = ?        | Done     | Done       | True  |
| 301 | Tensor<[1, 32, 32, 192]> self = , Tensor<[1, 32, 32, 192]> other = ?      | Done     | Done       | True  |
| 302 | Tensor<[1, 32, 32, 256]> self = , Tensor<[1, 32, 32, 256]> other = ?      | Done     | Done       | True  |
| 303 | Tensor<[1, 32, 49, 49]> self = , Tensor<[1, 32, 49, 49]> other = ?        | Done     | Done       | True  |
| 304 | Tensor<[1, 32, 56, 56]> self = , Tensor<[1, 32, 56, 56]> other = ?        | Done     | Done       | True  |
| 307 | Tensor<[1, 32, 64, 64]> self = , Tensor<[1, 32, 64, 64]> other = ?        | Done     | Done       | True  |
| 308 | Tensor<[1, 32, 75, 75]> self = , Tensor<[1, 32, 75, 75]> other = ?        | Done     | Done       | True  |
| 309 | Tensor<[1, 32, 95, 95]> self = , Tensor<[1, 32, 95, 95]> other = ?        | Done     | Done       | True  |
| 310 | Tensor<[1, 320, 14, 14]> self = , Tensor<[1, 320, 14, 14]> other = ?      | Done     | Done       | True  |
| 311 | Tensor<[1, 320, 64, 64]> self = , Tensor<[1, 320, 1, 1]> other = ?        | Unknown  | Done       | True  |
| 312 | Tensor<[1, 320, 64, 64]> self = , Tensor<[1, 320, 64, 64]> other = ?      | Unknown  | Done       | True  |
| 315 | Tensor<[1, 336, 14, 14]> self = , Tensor<[1, 336, 14, 14]> other = ?      | Done     | Done       | True  |
| 316 | Tensor<[1, 336, 56, 56]> self = , Tensor<[1, 336, 56, 56]> other = ?      | Done     | Done       | True  |
| 317 | Tensor<[1, 34, 28, 28]> self = , Tensor<[1, 34, 28, 28]> other = ?        | Done     | Done       | True  |
| 318 | Tensor<[1, 36, 28, 28]> self = , Tensor<[1, 36, 28, 28]> other = ?        | Done     | Done       | True  |
| 319 | Tensor<[1, 36, 56, 56]> self = , Tensor<[1, 36, 56, 56]> other = ?        | Done     | Done       | True  |
| 320 | Tensor<[1, 3712, 7, 7]> self = , Tensor<[1, 3712, 7, 7]> other = ?        | Done     | Done       | True  |
| 321 | Tensor<[1, 384, 35, 35]> self = , Tensor<[1, 384, 35, 35]> other = ?      | Done     | Done       | True  |
| 322 | Tensor<[1, 384, 8, 8]> self = , Tensor<[1, 384, 8, 8]> other = ?          | Done     | Done       | True  |
| 323 | Tensor<[1, 4, 12, 49, 49]> self = , Tensor<[1, 4, 1, 49, 49]> other = ?   | None     | Fallback   | True  |
| 324 | Tensor<[1, 4, 12, 64, 64]> self = , Tensor<[1, 4, 1, 64, 64]> other = ?   | None     | Fallback   | True  |
| 325 | Tensor<[1, 4, 16, 49, 49]> self = , Tensor<[1, 4, 1, 49, 49]> other = ?   | None     | Fallback   | True  |
| 326 | Tensor<[1, 4, 16, 64, 64]> self = , Tensor<[1, 4, 1, 64, 64]> other = ?   | None     | Fallback   | True  |
| 327 | Tensor<[1, 4, 768]> self = , Tensor<[1, 4, 768]> other = ?                | Unknown  | Done       | True  |
| 328 | Tensor<[1, 4, 768]> self = , Tensor<[4, 768]> other = ?                   | Unknown  | Done       | True  |
| 329 | Tensor<[1, 40, 14, 14]> self = , Tensor<[1, 40, 14, 14]> other = ?        | Done     | Done       | True  |
| 330 | Tensor<[1, 40, 28, 28]> self = , Tensor<[1, 40, 28, 28]> other = ?        | Done     | Done       | True  |
| 331 | Tensor<[1, 40, 30, 30]> self = , Tensor<[1, 40, 30, 30]> other = ?        | Done     | Done       | True  |
| 332 | Tensor<[1, 40, 40, 40]> self = , Tensor<[1, 40, 40, 40]> other = ?        | Done     | Done       | True  |
| 333 | Tensor<[1, 40, 56, 56]> self = , Tensor<[1, 40, 56, 56]> other = ?        | Done     | Done       | True  |
| 334 | Tensor<[1, 400, 7, 7]> self = , Tensor<[1, 400, 7, 7]> other = ?          | Done     | Done       | True  |
| 335 | Tensor<[1, 408, 14, 14]> self = , Tensor<[1, 408, 14, 14]> other = ?      | Done     | Done       | True  |
| 336 | Tensor<[1, 4096, 256]> self = , Tensor<[256]> other = ?                   | Done     | Done       | True  |
| 337 | Tensor<[1, 4096, 320]> self = , Tensor<[1, 4096, 320]> other = ?          | Unknown  | Done       | True  |
| 338 | Tensor<[1, 4096, 64]> self = , Tensor<[1, 4096, 64]> other = ?            | Done     | Done       | True  |
| 339 | Tensor<[1, 432, 14, 14]> self = , Tensor<[1, 432, 14, 14]> other = ?      | Done     | Done       | True  |
| 340 | Tensor<[1, 440, 7, 7]> self = , Tensor<[1, 440, 7, 7]> other = ?          | Done     | Done       | True  |
| 341 | Tensor<[1, 448, 28, 28]> self = , Tensor<[1, 448, 28, 28]> other = ?      | Done     | Done       | True  |
| 343 | Tensor<[1, 45, 3072]> self = , Tensor<[1, 45, 3072]> other = ?            | Unknown  | Done       | True  |
| 344 | Tensor<[1, 45, 768]> self = , Tensor<[1, 45, 768]> other = ?              | Unknown  | Done       | True  |
| 345 | Tensor<[1, 46, 28, 28]> self = , Tensor<[1, 46, 28, 28]> other = ?        | Done     | Done       | True  |
| 346 | Tensor<[1, 48, 14, 14]> self = , Tensor<[1, 48, 14, 14]> other = ?        | Done     | Done       | True  |
| 347 | Tensor<[1, 48, 33, 33]> self = , Tensor<[1, 48, 33, 33]> other = ?        | Done     | Done       | True  |
| 348 | Tensor<[1, 48, 38, 38]> self = , Tensor<[1, 48, 38, 38]> other = ?        | Done     | Done       | True  |
| 349 | Tensor<[1, 48, 56, 56]> self = , Tensor<[1, 48, 56, 56]> other = ?        | Done     | Done       | True  |
| 350 | Tensor<[1, 480, 14, 14]> self = , Tensor<[1, 480, 14, 14]> other = ?      | Done     | Done       | True  |
| 351 | Tensor<[1, 480, 7, 7]> self = , Tensor<[1, 480, 7, 7]> other = ?          | Done     | Done       | True  |
| 352 | Tensor<[1, 4800, 128]> self = , Tensor<[1, 4800, 128]> other = ?          | Done     | Done       | True  |
| 353 | Tensor<[1, 5, 1024]> self = , Tensor<[1, 5, 1024]> other = ?              | Unknown  | Done       | True  |
| 354 | Tensor<[1, 5, 16, 32]> self = , Tensor<[1, 5, 16, 32]> other = ?          | Unknown  | Done       | True  |
| 356 | Tensor<[1, 5, 4096]> self = , Tensor<[1, 5, 4096]> other = ?              | Unknown  | Done       | True  |
| 357 | Tensor<[1, 50, 1024]> self = , Tensor<[1, 50, 1024]> other = ?            | Done     | Done       | True  |
| 358 | Tensor<[1, 50, 3072]> self = , Tensor<[1, 50, 3072]> other = ?            | Done     | Done       | True  |
| 359 | Tensor<[1, 50, 768]> self = , Tensor<[1, 50, 768]> other = ?              | Done     | Done       | True  |
| 362 | Tensor<[1, 512, 100, 136]> self = , Tensor<[1, 512, 1, 1]> other = ?      | Unknown  | Done       | True  |
| 363 | Tensor<[1, 512, 100, 136]> self = , Tensor<[1, 512, 100, 136]> other = ?  | Unknown  | Done       | True  |
| 364 | Tensor<[1, 512, 14, 14]> self = , Tensor<[1, 512, 14, 14]> other = ?      | Done     | Done       | True  |
| 365 | Tensor<[1, 512, 23, 40]> self = , Tensor<[1, 512, 1, 1]> other = ?        | Done     | Done       | True  |
| 366 | Tensor<[1, 512, 25, 34]> self = , Tensor<[1, 512, 1, 1]> other = ?        | Unknown  | Done       | True  |
| 367 | Tensor<[1, 512, 28, 28]> self = , Tensor<[1, 512, 28, 28]> other = ?      | Done     | Done       | True  |
| 368 | Tensor<[1, 512, 32, 32]> self = , Tensor<[1, 512, 32, 32]> other = ?      | Done     | Done       | True  |
| 369 | Tensor<[1, 512, 45, 80]> self = , Tensor<[1, 512, 1, 1]> other = ?        | Done     | Done       | True  |
| 370 | Tensor<[1, 512, 50, 68]> self = , Tensor<[1, 512, 1, 1]> other = ?        | Unknown  | Done       | True  |
| 371 | Tensor<[1, 512, 7, 7]> self = , Tensor<[1, 512, 7, 7]> other = ?          | Done     | Done       | True  |
| 372 | Tensor<[1, 512, 8, 8]> self = , Tensor<[1, 512, 8, 8]> other = ?          | Done     | Done       | True  |
| 373 | Tensor<[1, 512, 90, 160]> self = , Tensor<[1, 512, 1, 1]> other = ?       | Done     | Done       | True  |
| 374 | Tensor<[1, 512, 90, 160]> self = , Tensor<[1, 512, 90, 160]> other = ?    | Done     | Done       | True  |
| 375 | Tensor<[1, 512]> self = , Tensor<[1, 512]> other = ?                      | Done     | Done       | True  |
| 376 | Tensor<[1, 528, 96, 96]> self = , Tensor<[1, 528, 96, 96]> other = ?      | Done     | Done       | True  |
| 377 | Tensor<[1, 56, 14, 14]> self = , Tensor<[1, 56, 14, 14]> other = ?        | Done     | Done       | True  |
| 378 | Tensor<[1, 56, 48, 48]> self = , Tensor<[1, 56, 48, 48]> other = ?        | Done     | Done       | True  |
| 379 | Tensor<[1, 56, 56, 128]> self = , Tensor<[1, 56, 56, 128]> other = ?      | Done     | Done       | True  |
| 380 | Tensor<[1, 56, 56, 96]> self = , Tensor<[1, 56, 56, 96]> other = ?        | Done     | Done       | True  |
| 381 | Tensor<[1, 576, 14, 14]> self = , Tensor<[1, 576, 14, 14]> other = ?      | Done     | Done       | True  |
| 382 | Tensor<[1, 58, 28, 28]> self = , Tensor<[1, 58, 28, 28]> other = ?        | Done     | Done       | True  |
| 383 | Tensor<[1, 59, 1024]> self = , Tensor<[1, 59, 1024]> other = ?            | Done     | Done       | True  |
| 385 | Tensor<[1, 6, 1, 15]> self = , Tensor<[1, 1, 1, 15]> other = ?            | Unknown  | Fallback   | True  |
| 386 | Tensor<[1, 6, 1, 15]> self = , Tensor<[1, 6, 1, 15]> other = ?            | Unknown  | Done       | True  |
| 387 | Tensor<[1, 6, 1, 17]> self = , Tensor<[1, 1, 1, 17]> other = ?            | Unknown  | Fallback   | True  |
| 388 | Tensor<[1, 6, 1, 17]> self = , Tensor<[1, 6, 1, 17]> other = ?            | Unknown  | Done       | True  |
| 389 | Tensor<[1, 6, 1, 1]> self = , Tensor<[1, 1, 1, 1]> other = ?              | Unknown  | Fallback   | True  |
| 390 | Tensor<[1, 6, 1, 1]> self = , Tensor<[1, 6, 1, 1]> other = ?              | Unknown  | Done       | True  |
| 391 | Tensor<[1, 6, 1, 2]> self = , Tensor<[1, 1, 1, 2]> other = ?              | Unknown  | Fallback   | True  |
| 392 | Tensor<[1, 6, 1, 2]> self = , Tensor<[1, 6, 1, 2]> other = ?              | Unknown  | Done       | True  |

| 395 | Tensor<[1, 6, 15, 15]> self = , Tensor<[1, 1, 1, 15]> other = ?           | Done     | Done       | True  |
| 396 | Tensor<[1, 6, 15, 15]> self = , Tensor<[1, 6, 15, 15]> other = ?          | Done     | Done       | True  |
| 397 | Tensor<[1, 60, 28, 28]> self = , Tensor<[1, 60, 28, 28]> other = ?        | Done     | Done       | True  |
| 400 | Tensor<[1, 64, 120, 160]> self = , Tensor<[1, 64, 120, 160]> other = ?    | Done     | Done       | True  |
| 401 | Tensor<[1, 64, 128, 128]> self = , Tensor<[1, 64, 128, 128]> other = ?    | Done     | Done       | True  |
| 402 | Tensor<[1, 64, 14, 14]> self = , Tensor<[1, 64, 14, 14]> other = ?        | Done     | Done       | True  |
| 403 | Tensor<[1, 64, 147, 147]> self = , Tensor<[1, 64, 147, 147]> other = ?    | Done     | Done       | True  |
| 404 | Tensor<[1, 64, 150, 150]> self = , Tensor<[1, 64, 150, 150]> other = ?    | Done     | Done       | True  |
| 405 | Tensor<[1, 64, 180, 320]> self = , Tensor<[1, 64, 1, 1]> other = ?        | Done     | Done       | True  |
| 406 | Tensor<[1, 64, 200, 272]> self = , Tensor<[1, 64, 1, 1]> other = ?        | Unknown  | Done       | True  |
| 407 | Tensor<[1, 64, 224, 224]> self = , Tensor<[1, 64, 224, 224]> other = ?    | Done     | Done       | True  |
| 408 | Tensor<[1, 64, 240, 320]> self = , Tensor<[1, 64, 240, 320]> other = ?    | Done     | Done       | True  |
| 409 | Tensor<[1, 64, 256, 256]> self = , Tensor<[1, 64, 256, 256]> other = ?    | Done     | Done       | True  |
| 410 | Tensor<[1, 64, 28, 28]> self = , Tensor<[1, 64, 28, 28]> other = ?        | Done     | Done       | True  |
| 411 | Tensor<[1, 64, 3, 49, 49]> self = , Tensor<[1, 64, 1, 49, 49]> other = ?  | None     | Fallback   | True  |
| 412 | Tensor<[1, 64, 3, 64, 64]> self = , Tensor<[1, 64, 1, 64, 64]> other = ?  | None     | Fallback   | True  |
| 413 | Tensor<[1, 64, 30, 40]> self = , Tensor<[1, 64, 30, 40]> other = ?        | Done     | Done       | True  |
| 414 | Tensor<[1, 64, 360, 640]> self = , Tensor<[1, 64, 1, 1]> other = ?        | Done     | Done       | True  |
| 415 | Tensor<[1, 64, 4, 49, 49]> self = , Tensor<[1, 64, 1, 49, 49]> other = ?  | None     | Fallback   | True  |
| 416 | Tensor<[1, 64, 4, 64, 64]> self = , Tensor<[1, 64, 1, 64, 64]> other = ?  | None     | Fallback   | True  |
| 417 | Tensor<[1, 64, 400, 544]> self = , Tensor<[1, 64, 1, 1]> other = ?        | Unknown  | Done       | True  |
| 418 | Tensor<[1, 64, 480, 640]> self = , Tensor<[1, 64, 480, 640]> other = ?    | Done     | Done       | True  |
| 419 | Tensor<[1, 64, 56, 56]> self = , Tensor<[1, 64, 56, 56]> other = ?        | Done     | Done       | True  |
| 420 | Tensor<[1, 64, 60, 80]> self = , Tensor<[1, 64, 60, 80]> other = ?        | Done     | Done       | True  |
| 421 | Tensor<[1, 64, 64, 128]> self = , Tensor<[1, 64, 64, 128]> other = ?      | Done     | Done       | True  |
| 422 | Tensor<[1, 64, 64, 64]> self = , Tensor<[1, 64, 64, 64]> other = ?        | Unknown  | Done       | True  |
| 423 | Tensor<[1, 64, 64, 96]> self = , Tensor<[1, 64, 64, 96]> other = ?        | Done     | Done       | True  |
| 424 | Tensor<[1, 64, 9, 9]> self = , Tensor<[1, 1, 1, 9]> other = ?             | Done     | Done       | True  |
| 425 | Tensor<[1, 640, 7, 7]> self = , Tensor<[1, 640, 7, 7]> other = ?          | Done     | Done       | True  |

| 430 | Tensor<[1, 672, 14, 14]> self = , Tensor<[1, 672, 14, 14]> other = ?      | Done     | Done       | True  |
| 431 | Tensor<[1, 672, 28, 28]> self = , Tensor<[1, 672, 28, 28]> other = ?      | Done     | Done       | True  |
| 432 | Tensor<[1, 672, 7, 7]> self = , Tensor<[1, 672, 7, 7]> other = ?          | Done     | Done       | True  |
| 433 | Tensor<[1, 68, 14, 14]> self = , Tensor<[1, 68, 14, 14]> other = ?        | Done     | Done       | True  |
| 434 | Tensor<[1, 696, 28, 28]> self = , Tensor<[1, 696, 28, 28]> other = ?      | Done     | Done       | True  |
| 436 | Tensor<[1, 7, 3072]> self = , Tensor<[1, 7, 3072]> other = ?              | Done     | Done       | True  |
| 437 | Tensor<[1, 7, 4544]> self = , Tensor<[1, 7, 4544]> other = ?              | Done     | Done       | True  |
| 438 | Tensor<[1, 7, 7, 1024]> self = , Tensor<[1, 7, 7, 1024]> other = ?        | Done     | Done       | True  |
| 439 | Tensor<[1, 7, 7, 768]> self = , Tensor<[1, 7, 7, 768]> other = ?          | Done     | Done       | True  |
| 440 | Tensor<[1, 7, 768]> self = , Tensor<[1, 7, 768]> other = ?                | Done     | Done       | True  |
| 441 | Tensor<[1, 71, 7, 64]> self = , Tensor<[1, 71, 7, 64]> other = ?          | Done     | Done       | True  |
| 442 | Tensor<[1, 71, 7, 7]> self = , Tensor<[7, 7]> other = ?                   | None     | Fallback   | True  |
| 443 | Tensor<[1, 72, 14, 14]> self = , Tensor<[1, 72, 14, 14]> other = ?        | Done     | Done       | True  |
| 444 | Tensor<[1, 72, 28, 28]> self = , Tensor<[1, 72, 28, 28]> other = ?        | Done     | Done       | True  |
| 445 | Tensor<[1, 72, 56, 56]> self = , Tensor<[1, 72, 56, 56]> other = ?        | Done     | Done       | True  |
| 446 | Tensor<[1, 720, 14, 14]> self = , Tensor<[1, 720, 14, 14]> other = ?      | Done     | Done       | True  |
| 447 | Tensor<[1, 728, 19, 19]> self = , Tensor<[1, 728, 19, 19]> other = ?      | Done     | Done       | True  |
| 448 | Tensor<[1, 728, 38, 38]> self = , Tensor<[1, 728, 38, 38]> other = ?      | Done     | Done       | True  |
| 449 | Tensor<[1, 7392, 12, 12]> self = , Tensor<[1, 7392, 12, 12]> other = ?    | Done     | Done       | True  |
| 450 | Tensor<[1, 768, 14, 14]> self = , Tensor<[1, 768, 14, 14]> other = ?      | Done     | Done       | True  |
| 451 | Tensor<[1, 768, 384]> self = , Tensor<[384]> other = ?                    | Done     | Done       | True  |
| 452 | Tensor<[1, 768, 7, 7]> self = , Tensor<[1, 768, 7, 7]> other = ?          | Done     | Done       | True  |
| 453 | Tensor<[1, 768]> self = , Tensor<[1, 768]> other = ?                      | Done     | Done       | True  |
| 454 | Tensor<[1, 78, 28, 28]> self = , Tensor<[1, 78, 28, 28]> other = ?        | Done     | Done       | True  |
| 455 | Tensor<[1, 784, 7, 7]> self = , Tensor<[1, 784, 7, 7]> other = ?          | Done     | Done       | True  |
| 456 | Tensor<[1, 8, 1, 10]> self = , Tensor<[1, 1, 1, 10]> other = ?            | Unknown  | Fallback   | True  |
| 457 | Tensor<[1, 8, 1, 10]> self = , Tensor<[1, 8, 1, 10]> other = ?            | Unknown  | Done       | True  |
| 458 | Tensor<[1, 8, 1, 1]> self = , Tensor<[1, 1, 1, 1]> other = ?              | Unknown  | Fallback   | True  |
| 459 | Tensor<[1, 8, 1, 1]> self = , Tensor<[1, 8, 1, 1]> other = ?              | Unknown  | Done       | True  |
| 460 | Tensor<[1, 8, 1, 2]> self = , Tensor<[1, 1, 1, 2]> other = ?              | Unknown  | Fallback   | True  |
| 461 | Tensor<[1, 8, 1, 2]> self = , Tensor<[1, 8, 1, 2]> other = ?              | Unknown  | Done       | True  |

| 464 | Tensor<[1, 8, 10, 10]> self = , Tensor<[1, 1, 1, 10]> other = ?           | Done     | Done       | True  |
| 465 | Tensor<[1, 8, 10, 10]> self = , Tensor<[1, 8, 10, 10]> other = ?          | Done     | Done       | True  |
| 466 | Tensor<[1, 8, 112, 112]> self = , Tensor<[1, 8, 112, 112]> other = ?      | Done     | Done       | True  |
| 467 | Tensor<[1, 8, 256, 2048]> self = , Tensor<[1, 1, 1, 2048]> other = ?      | Done     | Done       | True  |
| 468 | Tensor<[1, 8, 768]> self = , Tensor<[1, 8, 768]> other = ?                | Done     | Done       | True  |
| 469 | Tensor<[1, 8, 8, 1024]> self = , Tensor<[1, 8, 8, 1024]> other = ?        | Done     | Done       | True  |
| 470 | Tensor<[1, 8, 8, 768]> self = , Tensor<[1, 8, 8, 768]> other = ?          | Done     | Done       | True  |
| 471 | Tensor<[1, 80, 10, 10]> self = , Tensor<[1, 80, 10, 10]> other = ?        | Done     | Done       | True  |
| 472 | Tensor<[1, 80, 14, 14]> self = , Tensor<[1, 80, 14, 14]> other = ?        | Done     | Done       | True  |
| 473 | Tensor<[1, 80, 15, 15]> self = , Tensor<[1, 80, 15, 15]> other = ?        | Done     | Done       | True  |
| 474 | Tensor<[1, 80, 20, 20]> self = , Tensor<[1, 80, 20, 20]> other = ?        | Done     | Done       | True  |
| 475 | Tensor<[1, 80, 56, 56]> self = , Tensor<[1, 80, 56, 56]> other = ?        | Done     | Done       | True  |
| 476 | Tensor<[1, 80, 7, 7]> self = , Tensor<[1, 80, 7, 7]> other = ?            | Done     | Done       | True  |
| 477 | Tensor<[1, 88, 17, 17]> self = , Tensor<[1, 88, 17, 17]> other = ?        | Done     | Done       | True  |
| 478 | Tensor<[1, 888, 7, 7]> self = , Tensor<[1, 888, 7, 7]> other = ?          | Done     | Done       | True  |
| 479 | Tensor<[1, 896, 14, 14]> self = , Tensor<[1, 896, 14, 14]> other = ?      | Done     | Done       | True  |
| 480 | Tensor<[1, 9, 1024]> self = , Tensor<[1, 9, 1024]> other = ?              | None     | Fallback   | True  |
| 482 | Tensor<[1, 9, 128]> self = , Tensor<[1, 9, 128]> other = ?                | Done     | Done       | True  |
| 484 | Tensor<[1, 9, 16384]> self = , Tensor<[1, 9, 16384]> other = ?            | Done     | Done       | True  |
| 485 | Tensor<[1, 9, 2048]> self = , Tensor<[1, 9, 2048]> other = ?              | None     | Fallback   | True  |
| 487 | Tensor<[1, 9, 3072]> self = , Tensor<[1, 9, 3072]> other = ?              | Done     | Done       | True  |
| 489 | Tensor<[1, 9, 4096]> self = , Tensor<[1, 9, 4096]> other = ?              | None     | Fallback   | True  |
| 490 | Tensor<[1, 9, 768]> self = , Tensor<[1, 9, 768]> other = ?                | None     | Fallback   | True  |
| 492 | Tensor<[1, 9, 8192]> self = , Tensor<[1, 9, 8192]> other = ?              | Done     | Done       | True  |
| 493 | Tensor<[1, 912, 7, 7]> self = , Tensor<[1, 912, 7, 7]> other = ?          | Done     | Done       | True  |
| 494 | Tensor<[1, 92, 14, 14]> self = , Tensor<[1, 92, 14, 14]> other = ?        | Done     | Done       | True  |
| 495 | Tensor<[1, 96, 14, 14]> self = , Tensor<[1, 96, 14, 14]> other = ?        | Done     | Done       | True  |
| 496 | Tensor<[1, 96, 19, 19]> self = , Tensor<[1, 96, 19, 19]> other = ?        | Done     | Done       | True  |
| 497 | Tensor<[1, 96, 56, 56]> self = , Tensor<[1, 96, 56, 56]> other = ?        | Done     | Done       | True  |
| 498 | Tensor<[1, 96, 7, 7]> self = , Tensor<[1, 96, 7, 7]> other = ?            | Done     | Done       | True  |
| 499 | Tensor<[1, 960, 7, 7]> self = , Tensor<[1, 960, 7, 7]> other = ?          | Done     | Done       | True  |
| 500 | Tensor<[1, 98, 28, 28]> self = , Tensor<[1, 98, 28, 28]> other = ?        | Done     | Done       | True  |
| 508 | Tensor<[10, 10]> self = , Tensor<[10, 10]> other = ?                      | Done     | Done       | True  |
| 509 | Tensor<[100, 1, 256]> self = , Tensor<[100, 1, 256]> other = ?            | Done     | Done       | True  |
| 513 | Tensor<[12, 24, 24]> self = , Tensor<[12, 24, 24]> other = ?              | Done     | Done       | True  |
| 517 | Tensor<[13600, 1, 4]> self = , Tensor<[1, 9, 4]> other = ?                | Unknown  | Fallback   | True  |
| 522 | Tensor<[15, 15]> self = , Tensor<[15, 15]> other = ?                      | Done     | Done       | True  |
| 523 | Tensor<[16, 6, 49, 49]> self = , Tensor<[1, 6, 49, 49]> other = ?         | Done     | Done       | True  |
| 524 | Tensor<[16, 6, 64, 64]> self = , Tensor<[1, 6, 64, 64]> other = ?         | Done     | Done       | True  |
| 525 | Tensor<[16, 8, 49, 49]> self = , Tensor<[1, 8, 49, 49]> other = ?         | Done     | Done       | True  |
| 526 | Tensor<[16, 8, 64, 64]> self = , Tensor<[1, 8, 64, 64]> other = ?         | Done     | Done       | True  |
| 538 | Tensor<[2, 512]> self = , Tensor<[2, 512]> other = ?                      | Done     | Done       | True  |
| 539 | Tensor<[2, 7, 2048]> self = , Tensor<[2, 7, 2048]> other = ?              | Done     | Done       | True  |
| 540 | Tensor<[2, 7, 512]> self = , Tensor<[1, 7, 512]> other = ?                | None     | Fallback   | True  |
| 541 | Tensor<[2, 7, 512]> self = , Tensor<[2, 7, 512]> other = ?                | Done     | Done       | True  |
| 542 | Tensor<[2, 8, 7, 7]> self = , Tensor<[2, 1, 7, 7]> other = ?              | None     | Fallback   | True  |
| 543 | Tensor<[2048, 262]> self = , Tensor<[262]> other = ?                      | Done     | Done       | True  |
| 545 | Tensor<[221, 1, 4]> self = , Tensor<[1, 9, 4]> other = ?                  | Unknown  | Fallback   | True  |
| 549 | Tensor<[25, 4]> self = , Tensor<[25, 1]> other = ?                        | Unknown  | Done       | True  |
| 555 | Tensor<[3234, 1]> self = , Tensor<[3234, 1]> other = ?                    | Unknown  | Done       | True  |
| 556 | Tensor<[3234, 2]> self = , Tensor<[3234, 2]> other = ?                    | Done     | Done       | True  |
| 557 | Tensor<[3234]> self = , Tensor<[3234]> other = ?                          | Unknown  | Done       | True  |
| 559 | Tensor<[3400, 1, 4]> self = , Tensor<[1, 9, 4]> other = ?                 | Unknown  | Fallback   | True  |
| 562 | Tensor<[4, 12, 49, 49]> self = , Tensor<[1, 12, 49, 49]> other = ?        | Done     | Done       | True  |
| 563 | Tensor<[4, 12, 64, 64]> self = , Tensor<[1, 12, 64, 64]> other = ?        | Done     | Done       | True  |
| 564 | Tensor<[4, 16, 49, 49]> self = , Tensor<[1, 16, 49, 49]> other = ?        | Done     | Done       | True  |
| 565 | Tensor<[4, 16, 64, 64]> self = , Tensor<[1, 16, 64, 64]> other = ?        | Done     | Done       | True  |
| 571 | Tensor<[59, 1024]> self = , Tensor<[59, 1024]> other = ?                  | Done     | Done       | True  |
| 574 | Tensor<[63, 1, 4]> self = , Tensor<[1, 9, 4]> other = ?                   | Unknown  | Fallback   | True  |
| 575 | Tensor<[64, 3, 49, 49]> self = , Tensor<[1, 3, 49, 49]> other = ?         | Done     | Done       | True  |
| 576 | Tensor<[64, 3, 64, 64]> self = , Tensor<[1, 3, 64, 64]> other = ?         | Done     | Done       | True  |
| 577 | Tensor<[64, 4, 49, 49]> self = , Tensor<[1, 4, 49, 49]> other = ?         | Done     | Done       | True  |
| 578 | Tensor<[64, 4, 64, 64]> self = , Tensor<[1, 4, 64, 64]> other = ?         | Done     | Done       | True  |
| 585 | Tensor<[850, 1, 4]> self = , Tensor<[1, 9, 4]> other = ?                  | Unknown  | Fallback   | True  |
| 586 | Tensor<[8732, 1]> self = , Tensor<[8732, 1]> other = ?                    | Unknown  | Done       | True  |
| 587 | Tensor<[8732, 2]> self = , Tensor<[8732, 2]> other = ?                    | Done     | Done       | True  |
| 588 | Tensor<[8732]> self = , Tensor<[8732]> other = ?                          | Unknown  | Done       | True  |
| 589 | Tensor<[920, 1, 256]> self = , Tensor<[256]> other = ?                    | None     | Fallback   | True  |
| 590 | Tensor<[920, 1, 256]> self = , Tensor<[920, 1, 256]> other = ?            | Done     | Done       | True  |
"""

# |   0 | Tensor<[0, 1]> self = , Tensor<[0, 1]> other = ?                          | Unknown  | Fallback   | True  |
# |   1 | Tensor<[0]> self = , Tensor<[0]> other = ?                                | Unknown  | Fallback   | True  |
# | 592 | Tensor<[]> self = , Tensor<[]> other = ?                                  | Unknown  | Fallback   | True  |

# | 103 | Tensor<[1, 12, 1, s0 + 1]> self = , Tensor<[1, 1, 1, s0 + 1]> other = ?   | Unknown  | Unknown    | N/A   |
# | 104 | Tensor<[1, 12, 1, s0 + 1]> self = , Tensor<[1, 12, 1, s0 + 1]> other = ?  | Unknown  | Unknown    | N/A   |
# | 105 | Tensor<[1, 12, 1, s10 + 1]> self = , Tensor<[1, 1, 1, s10 + 1]> other = ? | Unknown  | Unknown    | N/A   |
# | 139 | Tensor<[1, 1280, s0, s1]> self = , Tensor<[1, 1280, 1, 1]> other = ?      | Unknown  | Unknown    | N/A   |
# | 140 | Tensor<[1, 1280, s0, s1]> self = , Tensor<[1, 1280, s0, s1]> other = ?    | Unknown  | Unknown    | N/A   |
# | 141 | Tensor<[1, 1280, s1, s2]> self = , Tensor<[1, 1280, 1, 1]> other = ?      | Unknown  | Unknown    | N/A   |
# | 142 | Tensor<[1, 1280, s1, s2]> self = , Tensor<[1, 1280, s1, s2]> other = ?    | Unknown  | Unknown    | N/A   |
# | 173 | Tensor<[1, 16, 1, s0 + 1]> self = , Tensor<[1, 1, 1, s0 + 1]> other = ?   | Unknown  | Unknown    | N/A   |
# | 174 | Tensor<[1, 16, 1, s0 + 1]> self = , Tensor<[1, 16, 1, s0 + 1]> other = ?  | Unknown  | Unknown    | N/A   |
# | 175 | Tensor<[1, 16, 1, s10 + 1]> self = , Tensor<[1, 1, 1, s10 + 1]> other = ? | Unknown  | Unknown    | N/A   |
# | 313 | Tensor<[1, 320, s1, s2]> self = , Tensor<[1, 320, 1, 1]> other = ?        | Unknown  | Unknown    | N/A   |
# | 314 | Tensor<[1, 320, s1, s2]> self = , Tensor<[1, 320, s1, s2]> other = ?      | Unknown  | Unknown    | N/A   |
# | 393 | Tensor<[1, 6, 1, s0 + 1]> self = , Tensor<[1, 1, 1, s0 + 1]> other = ?    | Unknown  | Unknown    | N/A   |
# | 394 | Tensor<[1, 6, 1, s0 + 1]> self = , Tensor<[1, 6, 1, s0 + 1]> other = ?    | Unknown  | Unknown    | N/A   |
# | 426 | Tensor<[1, 640, s0, s1]> self = , Tensor<[1, 640, 1, 1]> other = ?        | Unknown  | Unknown    | N/A   |
# | 427 | Tensor<[1, 640, s0, s1]> self = , Tensor<[1, 640, s0, s1]> other = ?      | Unknown  | Unknown    | N/A   |
# | 428 | Tensor<[1, 640, s1, s2]> self = , Tensor<[1, 640, 1, 1]> other = ?        | Unknown  | Unknown    | N/A   |
# | 429 | Tensor<[1, 640, s1, s2]> self = , Tensor<[1, 640, s1, s2]> other = ?      | Unknown  | Unknown    | N/A   |
# | 462 | Tensor<[1, 8, 1, s0 + 1]> self = , Tensor<[1, 1, 1, s0 + 1]> other = ?    | Unknown  | Unknown    | N/A   |
# | 463 | Tensor<[1, 8, 1, s0 + 1]> self = , Tensor<[1, 8, 1, s0 + 1]> other = ?    | Unknown  | Unknown    | N/A   |
# | 501 | Tensor<[1, s0*s1, 1280]> self = , Tensor<[1, s0*s1, 1280]> other = ?      | Unknown  | Unknown    | N/A   |
# | 502 | Tensor<[1, s0*s1, 640]> self = , Tensor<[1, s0*s1, 640]> other = ?        | Unknown  | Unknown    | N/A   |
# | 503 | Tensor<[1, s1*s2, 1280]> self = , Tensor<[1, s1*s2, 1280]> other = ?      | Unknown  | Unknown    | N/A   |
# | 504 | Tensor<[1, s1*s2, 320]> self = , Tensor<[1, s1*s2, 320]> other = ?        | Unknown  | Unknown    | N/A   |
# | 505 | Tensor<[1, s1*s2, 640]> self = , Tensor<[1, s1*s2, 640]> other = ?        | Unknown  | Unknown    | N/A   |


import re


def extract_tensor_shapes(data):
    tensor_shapes = []
    # Updated regex pattern to match single or multiple numbers inside Tensor brackets
    pattern = r"Tensor<\[(.*?)\]> self =\s*,\s*Tensor<\[(.*?)\]> other ="

    # Iterate through each line of the data
    for line in data.strip().split("\n"):
        match = re.search(pattern, line)
        if match:
            # Parse the shape strings into lists of integers
            self_shape = list(map(int, match.group(1).split(",")))  # Get self shape
            other_shape = list(map(int, match.group(2).split(",")))  # Get other shape
            tensor_shapes.append({"self": self_shape, "other": other_shape})

    return tensor_shapes


# Extract shapes
shapes = extract_tensor_shapes(data)
print(shapes)

# copy the shapes output and paste in file. change single quotes on self and other to double quotes and use black tool to align

# # Print the extracted shapes - self or other
# for shape in shapes:
#     vec = shape["self"]
#     print(f"{vec},")
