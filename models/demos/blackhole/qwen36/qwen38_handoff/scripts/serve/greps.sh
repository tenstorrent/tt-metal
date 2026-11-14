L=$1/server.log
echo "--Async sched:"; grep -n "Asynchronous scheduling is" $L | cut -c1-200
echo "--requested-but:"; grep -c "Async scheduling was requested, but TT model" $L
echo "--contract:"; grep -n "decode_input_update_contract\|legacy decode input reload contract" $L | cut -c1-250
echo "--TT submissions:"; grep "TT submissions:" $L | tail -3 | cut -c1-250; grep -c "TT submissions:" $L
echo "--TT async decode:"; grep "TT async decode:" $L | tail -3 | cut -c1-300
echo "--errors:"; grep -nE "Traceback|ERROR|TT_THROW" $L | head -10 | cut -c1-250; grep -cE "Traceback|ERROR|TT_THROW" $L
echo "--conv sync:"; grep -c "GDN conv-format sync" $L
echo "--trace_mode:"; grep -n "trace_mode=" $L | head -2 | cut -c1-250
