--- a/ttnn/transformer/ttnn_transformer_scaled_dot_product_attention_decode.cpp
+++ b/ttnn/transformer/ttnn_transformer_scaled_dot_product_attention_decode.cpp
@@ -387,7 +387,7 @@
     int dst_size = (fp32_dest_acc_en ? 4 : 8);
     int qk_out_subblock_h = min(PNHt, dst_size / qk_out_subblock_w);
-    int MUL_BCAST_GRANULARITY = qk_out_subblock_h;
+    int MUL_BCAST_GRANULARITY = min(qk_out_subblock_h, 32); // ensure MUL_BCAST_GRANULARITY <= 32

     // Rest of the file remains unchanged...
