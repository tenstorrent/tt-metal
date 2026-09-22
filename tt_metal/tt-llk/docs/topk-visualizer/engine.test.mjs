import assert from 'node:assert/strict';
import { buildTrace, makeInput, address } from './engine.js';

let checked = 0;
for (const preset of ['mixed', 'negative', 'descending']) {
  for (let seed = 1; seed <= (preset === 'descending' ? 1 : 100); seed++) {
    const input = makeInput(seed, preset);
    const trace = buildTrace(input);
    const expected = input.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value).slice(0, 32);
    assert.deepEqual(trace.output.map(({ value, index }) => ({ value, index })), expected);
    for (const [i, e] of trace.events.entries()) {
      // Index payloads must still point to the original value at every boundary.
      for (const item of [...e.dst, ...e.regs].filter(Boolean)) assert.equal(item.value, input[item.index]);
      assert.equal(new Set(e.dst.map(x => x.index)).size, 64, `DEST lost a candidate at event ${i}`);
      const regItems = e.regs.filter(Boolean);
      assert.equal(new Set(regItems.map(x => x.index)).size, regItems.length);
      if (e.kind === 'transpose') {
        const prev = trace.events[i - 1];
        for (let r = 0; r < 4; r++) for (let l = 0; l < 4; l++) {
          assert.equal(e.regs[r * 4 + l]?.index, prev.regs[l * 4 + r]?.index);
        }
      }
      if (e.runLength) {
        for (let start = 0; start < 64; start += e.runLength) {
          const run = e.dst.slice(start, start + e.runLength).map(x => x.value);
          const direction = (start / e.runLength) % 2 ? 1 : -1;
          assert.deepEqual(run, [...run].sort((a, b) => direction * (a - b)), `phase ${e.phase}`);
        }
      }
      if (e.chapter === 'merge' && e.selected) {
        const winners = e.dst.slice(0, 32).map(x => x.value);
        const losers = e.dst.slice(32).map(x => x.value);
        assert.ok(Math.min(...winners) > Math.max(...losers));
      }
    }
    checked++;
  }
}
assert.deepEqual([0, 4, 8, 12, 16, 32, 48].map(address), [0, 4, 8, 12, 32, 64, 96]);
assert.throws(() => buildTrace(Array(64).fill(1)), /distinct/);
assert.throws(() => buildTrace([1, 2]), /64/);
console.log(`PASS: ${checked} full traces; phase ordering, merge separation, transpose mapping, value/index provenance and final Top-32.`);
