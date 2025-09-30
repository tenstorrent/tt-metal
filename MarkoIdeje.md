# Marko ideje

## Grid sample generality

Trenutno grid sample assertuje sve zivo, issue https://github.com/tenstorrent/tt-metal/issues/28513

### 1. Wide reduction nije supported

    Compute vec podrzava. Ne bi trebalo da je problem napraviti da radi, ali ima par stvari na ovu temu kojima treba razmisliti, i u pool-u i u grid-sample-u:

    Nisam gledao nesto mnogo reader za pool, ali split reader radi tako sto ima 2 CB-ja, i alternira to na kome radi unpack_tilizeA_B
    Fora je sto kada je wide redukcija, cb se ne menja svaki put, nego za jedan output stick sve vreme je isti cb koji treba da se tilizuje.
    Recimo da ima cb0 i cb1, i ima 4 bloka za redukciju. U tom slucaju, na poolu bi islo

    Stick0: cb0 cb0 cb0 cb0
    Stick1: cb1 cb1 cb1 cb1
    Stick2: cb0 cb0 cb0 cb0

    ...

    Iskreno, ne znam da li je ovo optimalno za pool, nisam ga profilovao kada jeste wide redukcija. Btw, grid sample uvek alternira, i kada extenduje kanale,
    pa se dobije recimo za extend_factor=7 nesto ovako

    OutputStick0: cb0 cb1 cb0 cb1 cb0 cb1 cb0
    OutputStick1: cb1 cb0 cb1 cb0 cb1 cb0 cb1

    (ovo je razlog zasto grid_sample_sharded reader kernel ima jedan loop, da bi podela ovoga za split_reader bila laksa. Interleaved ima dva loopa)

    ...

    Uglavnom, osim ove konsideracije za pool, za grid sample je zavrzlama sledece. Postoje dve opcije:

    1. Downloadovati ceo remote stick u neki cb, onda raditi n_blocks = divup(num_channels, (32 * 8)) lokalnih transfera, ukupno 1 + n_blocks
    2. Raditi n_blocks remote transfera

    Iskreno ne znam sta je bolje od ova dva. Pool ima halo i tu je sve lokalno, kontam radi opciju 2. samo sto transferi nisu remote nego local.
    Ovo alterniranje za wide redukciju da ide po sticku je coded u kernelu (isti je skalar CB), ali to nam za extendovanje kanala na GS i treba,
    jer su to tehnicki drugi output stickovi samo se radi concat u L1

### 2. Kanali required na 32

    Slicna situacija kao proslo, compute podrzava samo niko nije trazio takve shape-ove pa nisam napravio.
    Ovde ne znam koliko ima zavrzlama, kontam ne toliko. Kada sam rebase-ovao na ovu promenu sam stavio da mi page-size out_cb-ja ne bude TILE_WIDTH nego FACE_WIDTH. Ne znam koje sve izmene su neophodne u readeru

### 3. Tiled input/output

    Tiled input slika - nemam pojma, mozda u nekom univerzumu moze da se namesti da radi nesto, ali kontam ne
    Tiled grid - morao bi da se ili radi untilize, ili da se radi ono direktno adresiranje u tile. To mozda i zapravo nije toliko lose, iskreno
    ako je uradivo i save-uje perf ja jesam za to. Bilo bi jako nezgodno doduse jer recimo precomputed grid, grid batching 7, ukupno 42 datum-a po redu.
    Za 6. stick, 30. i 31. kolona imaju indexe, a 32., 33., 34. i 35. (znaci u drugom TILE-u) ima weightove.

    ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp

    Ovaj kernel se bavi time. Mozda je i ok upustiti se u ovo
    Tiled output - ono Slavkovo. Los perf ali skoro sigurno moze da se namesti da radi.
    Ne znam koja su ogranicenja za visinu shard-a na tome, kontam da je na 32, ovde isto treba razlisliti i o visina shard-a a i o podeli posla u interleaved

### 4. Onaj bag sa tilize - untilize
    mstaletovic/GridSamplePrecomputeChanges

    Na tom branchu bleje WIP promene za to

    Uglavnom, postojala su dva baga
    1. Los alignment - samo kada je interleaved radim da je stick_size = round_up(padded_shape[-1] * datum_size(), alignment), treba da se radi u oba slucaja. Ovo je pravilo probleme i za precomputed_grid = true i za precomputed_grid = false. Takodje, nisam siguran da li je pametno da bude padded_shape, mozda bi bilo bolje shard_shape[1], na HS vljd nije razlika ali mozda je vise obvious
    2. Encodovanje int16 as bfloat16 - problem nastaje recimo kada postoji intidzer jednak 1, bitski 0000 0000 0000 0001, tj exp0 i mantisa!=1, subnormal broj. Oni izgleda da mogu da zive u L1 i u DRAM-u, ali kada dodatku src/dest/sta god izgleda da ih on rounduje na nulu
    Umesto toga koristimo uint16 za enkodovanje svega, to radi.

    Mislim da Marko i Natasa rade sa ovom promenom.

    Testovi i dalje failuju doduse u ovoj situaciji (i to je zasto nisam hteo ovo jos na PR), ako grid dimenzije nisu flattened tj (1, 1, N_out x H_out_ x W_out, nesto), kontam jer kada nije flattened poslednje dve dimenziju su padded a ne kao po NHW, a to nije handlovano.

### 5. Float32 / TF32

    Mozda ce morati da se implementira jer preciznost myb nije dovoljno dobra za OFT u BFP16. Ovo bi impliciralo da se podrzi Float32 / TF32 na FPU za samu redukciju, ne ovo sto sam ja radio samo za grid ali da na kraju ode na BFP16 svakako.

    Ovo ima impact-a i sa DM i sa compute strane.

    Na DM strani, ovo je u teoriji duplo vise svega, ali iskreno mislim da nece praviti toliki problem, jer su transferi i dalje u zoni gde moze da se dobije na bandwidthu samim povecavanjem velicine transfera, pa mozda DM zap bude slicne brzine, a mozda je i wishful thinking.

    Sa compute strane ne znam kolika je razlika. Ako je tilize duplo sporiji, srecno nam bilo. Mislim da je ostatak stvari not so bad, to jest da ta dodatna 3 bita mantise na FPU-ju ne pravi toliko problem za redukciju perf - wise. Takodje, kontam zelimo fp32 accumulation u ovom slucaju, pa pravljenje ovoga implicira wide redukciju za Markov shape

    Sad kad razmislim ponovo, pisuci ovo za compute mi je sad palo na pamet da transferi, zbog wide redukcije, mozda i nece biti veci


## Grid sample API

Verovatno bi trebalo napraviti neke promene na API-ju.

### Kako primati input i grid

Za sada su oba N, H, W, C, i iz toga se extractuju neke informacije. Ovo je dobro i torch-like, ali npr malo je nezgrapno za ovo gore sto sam pomenuo, sto ima bug kada grid nije flattened. Moglo bi biti tako da se salje (1, 1, NHW, C) i (1, 1, NHoutWout, grid_whatever). Mozda najbolje supportovati oba. Kontam pool i conv su kompleksniji pa ima smisla da imaju vise loaded API, grid sample mozda bolje da keep it simple

### ENUM umesto nekih stvari

Znam da Artem ne voli bool-ove u API-ju, trenutno ih grid sample ima. Mozda prevesti to u neke enume za precomputed grid i za batching.

## Grid sample perf

Trenutno, situacija stoji ovako, za precomputed_grid = True i L1 interleaved:

Na BH smo tilize bound za veci broj kanala (224, 256), bilo da je split reader ukljucen ili iskljucen (i iz nekog razloga ga split reader uspori. Slicna situaciji postoji i pool-u. Nije mi uopste jasno sto postoji taj problem, maybe vredi istraziti). Za srednji broj kanala, bez split readera smo reader bound, sa split readerom smo tilize, za jako male brojeve kanala (32/64) mislim da je i dalje reader bound cak i sa split readerom


Na WH je slicna situacija samo sto su granice malo pomerene, jer je tilize bolji. Za 256 WH je bez split readera i dalje DM bound, tek sa split readerom postaje tilize bound. Nisam dugo merio, ali slicno je kontam i za srednji broj kanala, pa onda u nekom trenutku postaje opet DM cak i sa 2 reader-a.

Za DRAM interleaved nisam bas gledao, kontam da je manje vise uvek reader bound. Split reader iskljucen po defaultu zato sto je jedan noc mnogo brzi u citanju nego drugi.

### Ideje za tilize bound

### Full matmul

Umesto da se radi redukcija, radi se neki (sparse) matmul. Ovo se svodi na to da levi operand bude neka matrica weightova, a desni neki stickovi. MATMUL out = B*A

B =
[
    w00 w01 w02 w03  0  0  0  0  ...
      0   0   0   0  w10 w11 w12 w13    0   0   0   0
      0   0   0   0    0   0   0   0  w20 w21 w22 w23 ...

      ...
]

A =
[
    stick00
    stick01
    stick02
    stick03
    stick10
    stick11
    stick12
    stick13

    ...
],

gde se prvi indexi ticu output-a a drugi inputa.

Recimo da ima n_tiles_c tile-ova u sirinu za matricu A, a u visinu recimo da se stavi 4 tile-a. Ta 4 tile-a su inner dim. Matrica B bi bila 1_tile x 4_tile-a. Znaci islo bi

(1 x 4) @ (4 x n_tiles_c) = 1 x n_tiles_c

Ovo je u Markovom case-u jako srecno zato sto n_tiles_c = 8, upotrebi se ceo dest i to je to. Za vise kanala opet mora nekako da se zonglira. Ovo bi nam dalo native output da je TILEd, a ne row major. Mislim da se channel extending ne mesa lepo sa ovim uopste, jer se tamo konkateniraju individualni redovi. Jedna ideja koja mi se ne svidja:

TILE0 TILE1 TILE2 TILE3 TILE4 TILE5 TILE6 TILE7, gde ih ovde ima 32 u visinu, 32x256. U ovih 32, NEMA onih koji se koncatuju jedan na drugi, nego u sledecoj turi od 32 u visinu se uzmu oni koji treba da se contatuju, pa tako 7 tura. Problem sa ovim bi bilo to sto bi unit za paralelizaciju bio jako velik, pa bi podela po korovima bila jako losa.

Nadam se da ima neka bolja ideja za ovo. Mozda bi i mogli nekako da idu u istu turu ti sto treba da se extenduju, not sure kako




Treba razmisliti i kako popuniti matricu weightova B. Moze da bude spremna u DRAM-u, vec tile-ovana i da se tako streamuje, ali to zauzima dosta i ne skalira se na non precomputed grid nikako.


Druga opcija je da RISC ide i popunjava tu matricu, a da se na pocetku dovuku nule. Ovo moze jer je sparse matrica pa vljd nije toliko sporo, treba razmisliti o ova dva.

### Tilize odvejen umesto unpack_tilizeA_B_block

Ono sto sam zapoceo, dodacu branch posle. Uglavnom, ideja je da podaci dovode do srcA i srcB ne preko unpack_tilizeA_B_block, vec preko tilize_block, onda unpackA_B. Onda bi ovi iz LLK-ja potencijalno mogli da uzmu i dodaju support za face_r_dim na tilize_block, sto bi nadamo se ubrzalo tilize. Problem je u tome sto tilize_block koristi i reconfiguruje sva 3 threada, pa ce initi potencijalno biti preskupi jer ce ih biti mnogo.

Za sada sam ja uspeo da ne hanguje, ali pcc nije dobar. Izmerio sam perf na tome, isto nije dobar. Plan mi je bio da prvo namestim dobar PCC, pa onda da polako vidim sta od inita i tensix_sync-ova mogu da uklonim iz koda a da on i dalje radi.

### Redukcija u jednom TILE-u preko matmula

Recimo da je tile 8x8, i da je stick duzine 16. Onda 4 sticka mozemo da stavimo u jedan tile na sledeci nacin

A =
[
s00 s01 s02 s03 s04 s05 s06 s07
s08 s09 s0a s0b s0c s0d s0e s0f
s10 s11 s12 s13 s14 s15 s16 s17
s18 s19 s1a s1b s1c s1d s1e s1f
s20 s21 s22 s23 s24 s25 s26 s27
s28 s29 s2a s2b s2c s2d s2e s2f
s30 s31 s32 s33 s34 s35 s36 s37
s38 s39 s3a s3b s3c s3d s3e s3f
]

Kada bismo matmulovali to sa matricom

B =
[
w0  0 w1  0 w2  0 w3  0
0  w1  0 w2  0 w3  0 w4
...
]

C = B @ A, dobio bi se dobar rezultat, samo sto bi bio spakovan u 2 reda u destu. Za slucaj 256 kanala, ovo bi se spakovalo u jedan tile tako sto bi svaki stick zauzimao 8 u visinu, 4 bi zauzela tacno jedan tile. Onda bi matrica B imala 8 visinu, (svaki red bi bio 4 weighta i 7 * 4 = 28 nula), i rezultat je 8 redova u destu.

Treba razmisliti da li ovo radi za kada nije tile aligned. Takodje, kontam ne radi za kanale vece od 256. Skroz se slucajno dobilo da je isto ogranicenje i za ovo i za wide redukciju (tile 32, bilinear 4 sticka uzima, 32 / 4 = 8).

Ovo ima velikih generality problema, mada mislim da nije toliko tesko da se implementira a mislim da ni perf ne bi bio tako los, najveci problem vrv popunjavanje matrice B, potencijalno i initi.

### Reader bound ideje

### HS + coalesced reads

Ovde bi ideja bila da se slika height sharduje, i onda bi dva sticka koja treba da se citaju bila kontinualna u memoriji, pa bi mogla da se urade preko jednog transfera.

Odmah da kazem, mislim da ova ideja ipak nije nesto. Na osnovu nove dokumentacije https://github.com/tenstorrent/tt-low-level-documentation/ ispada da je broj transakcija dosta vazan faktor. Nema primer za 2 requesta, ali ima za 1 i 4, i ispada da nije neka razlika da li ces 4 manja i 1 veci.

Svakako, ima dva path-a za ovo.

1. Bez halo-a

Nije zagarantovano da su dva susedna sticka na istom koru, treba da se uradi check, mozda risc overhead

2. Sa halo-om

Garantuje da ce ta dva sticka biti na istom koru, potencijalna zavrzlama sa tim sa kog indexa treba da citas.

Takodje, ovo ne radi za stickove vece od 256, jer moraju iz vise da se citaju svakako

### Kesiranje u seperate data strukturi

Moguce je da se stickovi stave u neki data structure za kesiranje (recimo prosta mapa index -> stick) gde bleje prethodno procitani stickovi. Treba razmisliti o velicini cache-a, i o tome koliko bi ovo pomoglo.

Simplest implementation:

Napravi se mapa index -> stick.

Ako nije cache hit, downloadujes grid u cache, onda pomeris iz cache u cb.

Ako jeste cache hit, samo iz lokalnog cachea u cb

Svi transferi preko noca naravno

Overhead - cache miss znaci dva read-a, plus uvek ima ifova.

Benefit - transferi lokalni

Sa split readerom mora neka cache invalidation skalamerija

### Kesiranje u cb-ju

Ako treba da se interpoliraju ista 4 sticka kao u prosloj redukciji, probas da samo da ne downloadujes sticove uopste, nego oni vec stoje u cb-ju. Jedan prost if bi ovde radio, da nema double bufferinga. Double buffering je trenutno bitan za perf jako, ne znam koji je najbolji nacin da se napravi da se ova dva featura bolje slazu.

Ovo se ne mesa sa MATMUL idejom manje vise nikako, jer je verovatnoca da ces moci da resujujes input cb jako jako mala.


### Megaop

Vec sam dosta objasnio o tome. Implementacija i nije toliko teska, ali je problem odrzavanje svega toga. Ako bas bude problem sa BOSom onda ima smisla, inace verovatno ne.

### Precomputed_grid = False

U ovom slucaju moze svasta da se radi. Postoji ideja da neka verzija precoputed_grid = True postane "mikro_op", koji skuplja stickove i radi bilinearnu interpolaciju, a da precomputovanje grida bude pre toga nekako. Mozda koristi i za deformable conv ili tako nesto.

Sto se tice same kalkulacije, ja mislim da je najbolje da ona ode na SFPU. Svakako mora sfpu barem za nesto, jer postoji floor i typecast, pa kad vec mora mislim da je bolje da sve budu tu. Takodje, potencijalno je neophodna preciznost od fp32. Iskreno ne znam sta sve moze da se uradi na sfpu, ali znam da ove standardne eltwise moze da radi, isto i floor i typecast, tkd valjda moze da uradi sve sto nam treba. Treba razmisliti kako redjati to u tile-ove, da li nam je mozda tile-ovan grid bolji i tako to.
