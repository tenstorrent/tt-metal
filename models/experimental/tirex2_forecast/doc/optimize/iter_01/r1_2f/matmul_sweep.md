# R1-2f matmul sweep (M = 131072, HiFi4, fp32 acc; median of 10)

| role | config | ms | pcc |
|---|---|---|---|
| mlstm_qkv | linear default | 5.71 | 1.0 |
| mlstm_qkv | minimal M8 K8 N8 sub2x2 | 2.53 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N8 sub1x4 | 2.66 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N8 sub4x1 | 2.53 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K8 N4 sub2x2 | 2.61 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N4 sub1x4 | 2.60 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N4 sub4x1 | 2.59 | 0.9999999997600746 |
| mlstm_qkv | minimal M4 K8 N8 sub2x2 | 2.92 | 0.9999999997600746 |
| mlstm_qkv | minimal M4 K8 N8 sub1x4 | 2.89 | 0.9999999997600746 |
| mlstm_qkv | minimal M4 K8 N8 sub4x1 | 2.86 | 0.9999999997600746 |
| mlstm_qkv | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | linear default | 1.99 | 1.0 |
| mlstm_o_out | minimal M8 K8 N8 sub2x2 | 1.03 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N8 sub1x4 | 1.00 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N8 sub4x1 | 0.99 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K8 N4 sub2x2 | 1.04 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N4 sub1x4 | 1.04 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N4 sub4x1 | 1.04 | 0.9999999997568747 |
| mlstm_o_out | minimal M4 K8 N8 sub2x2 | 0.97 | 0.9999999997568747 |
| mlstm_o_out | minimal M4 K8 N8 sub1x4 | 0.93 | 0.9999999997568747 |
| mlstm_o_out | minimal M4 K8 N8 sub4x1 | 0.92 | 0.9999999997568747 |
| mlstm_o_out | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | linear default | 6.67 | 1.0 |
| ffn_down | minimal M8 K8 N8 sub2x2 | 2.73 | 0.9999999996389548 |
| ffn_down | minimal M8 K8 N8 sub1x4 | 2.65 | 0.9999999996389548 |
| ffn_down | minimal M8 K8 N8 sub4x1 | 2.66 | 0.9999999996389548 |
| ffn_down | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K8 N4 sub2x2 | 2.60 | 0.9999999996389548 |
| ffn_down | minimal M8 K8 N4 sub1x4 | 2.59 | 0.9999999996389548 |
| ffn_down | minimal M8 K8 N4 sub4x1 | 2.57 | 0.9999999996389548 |
| ffn_down | minimal M4 K8 N8 sub2x2 | 2.64 | 0.9999999996389548 |
| ffn_down | minimal M4 K8 N8 sub1x4 | 2.65 | 0.9999999996389548 |
| ffn_down | minimal M4 K8 N8 sub4x1 | 2.66 | 0.9999999996389548 |
| ffn_down | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K16 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K16 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K16 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | linear default | 10.79 | 1.0 |
| w_in | minimal M8 K8 N8 sub2x2 | 5.63 | 0.9999999759408288 |
| w_in | minimal M8 K8 N8 sub1x4 | 5.51 | 0.9999999759408288 |
| w_in | minimal M8 K8 N8 sub4x1 | 5.37 | 0.9999999759408288 |
| w_in | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M8 K8 N4 sub2x2 | 5.59 | 0.9999999759408288 |
| w_in | minimal M8 K8 N4 sub1x4 | 5.55 | 0.9999999759408288 |
| w_in | minimal M8 K8 N4 sub4x1 | 5.59 | 0.9999999759408288 |
| w_in | minimal M4 K8 N8 sub2x2 | 6.48 | 0.9999999759408288 |
| w_in | minimal M4 K8 N8 sub1x4 | 6.45 | 0.9999999759408288 |
| w_in | minimal M4 K8 N8 sub4x1 | 6.13 | 0.9999999759408288 |
| w_in | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M8 K16 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M8 K16 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| w_in | minimal M8 K16 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| embed_hidden | linear default | 0.93 | 1.0 |
| embed_hidden | minimal M8 K2 N8 sub2x2 | 0.71 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N8 sub1x4 | 0.72 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N8 sub4x1 | 0.69 | 0.9999999999995944 |
| embed_hidden | minimal M16 K2 N8 sub2x2 | 0.69 | 0.9999999999995944 |
| embed_hidden | minimal M16 K2 N8 sub1x4 | 0.71 | 0.9999999999995944 |
| embed_hidden | minimal M16 K2 N8 sub4x1 | 0.68 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N4 sub2x2 | 0.69 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N4 sub1x4 | 0.66 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N4 sub4x1 | 0.68 | 0.9999999999995944 |
| embed_hidden | minimal M4 K2 N8 sub2x2 | 0.61 | 0.9999999999995944 |
| embed_hidden | minimal M4 K2 N8 sub1x4 | 0.66 | 0.9999999999995944 |
| embed_hidden | minimal M4 K2 N8 sub4x1 | 0.69 | 0.9999999999995944 |
| embed_hidden | minimal M16 K2 N4 sub2x2 | 0.71 | 0.9999999999995944 |
| embed_hidden | minimal M16 K2 N4 sub1x4 | 0.68 | 0.9999999999995944 |
| embed_hidden | minimal M16 K2 N4 sub4x1 | 0.70 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N16 sub2x2 | 0.76 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N16 sub1x4 | 0.73 | 0.9999999999995944 |
| embed_hidden | minimal M8 K2 N16 sub4x1 | 0.76 | 0.9999999999995944 |
| head_out | linear default | 1.17 | 1.0 |
| head_out | minimal M8 K8 N8 sub2x2 | 0.63 | 0.9999999997420717 |
| head_out | minimal M8 K8 N8 sub1x4 | 0.58 | 0.9999999997420717 |
| head_out | minimal M8 K8 N8 sub4x1 | 0.58 | 0.9999999997420717 |
| head_out | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M8 K8 N4 sub2x2 | 0.57 | 0.9999999997420717 |
| head_out | minimal M8 K8 N4 sub1x4 | 0.57 | 0.9999999997420717 |
| head_out | minimal M8 K8 N4 sub4x1 | 0.57 | 0.9999999997420717 |
| head_out | minimal M4 K8 N8 sub2x2 | 0.62 | 0.9999999997420717 |
| head_out | minimal M4 K8 N8 sub1x4 | 0.61 | 0.9999999997420717 |
| head_out | minimal M4 K8 N8 sub4x1 | 0.61 | 0.9999999997420717 |
| head_out | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| head_out | minimal M8 K16 N9 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
